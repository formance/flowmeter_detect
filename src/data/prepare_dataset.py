"""Build a three-class DeepLabV3+ dataset from manually reviewed editor samples."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageOps

PROJECT = Path(__file__).resolve().parents[2]
SCENES = ('scene01_part', 'scene02', 'scene03')
SPLITS = ('train', 'val', 'test')
CLASSES = {'background': 0, 'pointer': 1, 'scale_tick': 2}
TICK_LINE_WIDTH = 1


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def write_scene_manifests(dataset):
    report=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    for scene in SCENES:
        rows=[row for row in report['samples'] if row['scene']==scene]
        pixels=Counter()
        for row in rows:
            if row['split']=='train':pixels.update({int(key):value for key,value in row['pixels'].items()})
        subreport={**report,'samples':rows,'counts':report['scenes'][scene]['counts'],
                   'scenes':{scene:report['scenes'][scene]},'train_class_pixels':dict(pixels),
                   'excluded_deleted':[name for name in report['excluded_deleted'] if name.startswith(scene+'_')]}
        write_json(dataset/'by_scene'/scene/'manifest.json',subreport)
        write_json(dataset/'by_scene'/scene/'classes.json',report['classes'])


def annotation_mask(annotation, size):
    if size != (annotation['imageWidth'], annotation['imageHeight']):
        raise ValueError('Image and annotation dimensions differ')
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    present = set()
    for label, kind, value in [('scale_tick', 'line', 2), ('pointer', 'polygon', 1)]:
        for shape in annotation['shapes']:
            if shape['label'] != label:
                continue
            points = shape['points']
            if shape['shape_type'] != kind or (len(points) != 2 if kind == 'line' else len(points) < 3):
                raise ValueError(f'Invalid {label} geometry')
            if any(len(p) != 2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in p)
                   or not 0 <= p[0] <= size[0]-1 or not 0 <= p[1] <= size[1]-1 for p in points):
                raise ValueError('Invalid annotation coordinates')
            if kind == 'line':
                if shape.get('line_width') != TICK_LINE_WIDTH:
                    raise ValueError(f'Tick line widths must already be fixed at {TICK_LINE_WIDTH}')
                draw.line([tuple(p) for p in points], fill=value, width=TICK_LINE_WIDTH)
                radius = (TICK_LINE_WIDTH-1) / 2
                for x, y in points:
                    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=value)
            else:
                draw.polygon([tuple(p) for p in points], fill=value)
            present.add(label)
    counts = Counter(mask.tobytes())
    if present != {'pointer', 'scale_tick'} or not counts[1] or not counts[2]:
        raise ValueError('Both foreground classes must contain pixels')
    return mask, counts


def features(image):
    gray = ImageOps.grayscale(image)
    tiny = np.asarray(gray.resize((32,32),Image.Resampling.LANCZOS),np.float64)
    indices = np.arange(32)
    dct = np.cos(np.pi*(indices[None,:]+.5)*indices[:,None]/32)
    coefficients = (dct @ tiny @ dct.T)[:8,:8].ravel()[1:]
    bits = coefficients > np.median(coefficients)
    vector = np.asarray(gray.resize((48,48),Image.Resampling.BILINEAR),np.float64).ravel()
    vector -= vector.mean()
    vector /= max(float(np.linalg.norm(vector)),1e-12)
    return bits,vector


def quota(count):
    raw=np.array([.7,.2,.1])*count
    numbers=np.floor(raw).astype(int)
    order=sorted(range(3),key=lambda k:(-(raw[k]-numbers[k]),k))
    for index in order[:count-int(numbers.sum())]:numbers[index]+=1
    return numbers.tolist()


def feasible_partition(groups, target, rng):
    """Exact 7:2:1 rounded counts while identical-image groups stay intact."""
    order=list(range(len(groups)));rng.shuffle(order)
    paths={(0,0):[]}
    for index in order:
        size=len(groups[index]);following={}
        choices=[0,1,2];rng.shuffle(choices)
        items=list(paths.items());rng.shuffle(items)
        for (val,test),path in items:
            for split in choices:
                nv=val+(size if split==1 else 0);nt=test+(size if split==2 else 0)
                if nv<=target[1] and nt<=target[2] and (nv,nt) not in following:
                    following[(nv,nt)]=path+[(index,split)]
        paths=following
    chosen=paths.get((target[1],target[2]))
    if chosen is None:
        raise ValueError('Exact split sizes cannot preserve duplicate groups; choose a new output policy')
    result=np.zeros(len(groups),int)
    for index,split in chosen:result[index]=split
    return result


def spread_scene(records, rng, attempts):
    groups_by_key=defaultdict(list)
    for record in records:groups_by_key[record['group_id']].append(record)
    groups=list(groups_by_key.values());target=quota(len(records))
    n=len(groups);weights=np.zeros((n,n),np.float64)
    relationships=[]
    for i in range(n):
        for j in range(i+1,n):
            nearest=min(abs(a['sequence']-b['sequence']) for a in groups[i] for b in groups[j])
            bits_a,vector_a=groups[i][0]['features'];bits_b,vector_b=groups[j][0]['features']
            hamming=int(np.count_nonzero(bits_a!=bits_b));cosine=float(vector_a@vector_b)
            adjacent=nearest<=2
            similar=hamming<=10 or cosine>=.94
            weight=(3/nearest if adjacent and nearest else 0)+(max(0,11-hamming)/5 if similar else 0)
            weights[i,j]=weights[j,i]=weight
            if adjacent or similar:relationships.append((i,j,adjacent,similar,hamming,cosine))
    sizes=np.array([len(group) for group in groups])
    best=None;best_score=float('inf');first_score=None
    for _ in range(attempts):
        assigned=feasible_partition(groups,target,rng)
        initial=float(np.triu(weights*(assigned[:,None]==assigned[None,:]),1).sum())
        if first_score is None:first_score=initial
        # Pair swaps preserve exact per-scene quotas and duplicate groups.
        for iteration in range(50):
            costs=np.column_stack([weights[:,assigned==split].sum(axis=1) for split in range(3)])
            improvement=0.;pair=None
            for i in range(n):
                for j in range(i+1,n):
                    if sizes[i]!=sizes[j] or assigned[i]==assigned[j]:continue
                    a,b=assigned[i],assigned[j]
                    delta=costs[i,b]-costs[i,a]+costs[j,a]-costs[j,b]-2*weights[i,j]
                    if delta<improvement-1e-10:improvement,pair=delta,(i,j)
            if pair is None:break
            i,j=pair;assigned[i],assigned[j]=assigned[j],assigned[i]
        score=float(np.triu(weights*(assigned[:,None]==assigned[None,:]),1).sum())
        if score<best_score:best,best_score=assigned.copy(),score
    assignment={record['id']:SPLITS[int(best[index])] for index,group in enumerate(groups) for record in group}
    report={'counts':dict(zip(SPLITS,target)),'duplicate_groups':[[r['id'] for r in group] for group in groups if len(group)>1],
            'similarity_objective_before':round(first_score,4),'similarity_objective_after':round(best_score,4),
            'adjacent_group_pairs':sum(a for _,_,a,_,_,_ in relationships),
            'adjacent_pairs_in_different_splits':int(sum(a and best[i]!=best[j] for i,j,a,_,_,_ in relationships)),
            'similar_group_pairs':sum(s for _,_,_,s,_,_ in relationships),
            'similar_pairs_in_different_splits':int(sum(s and best[i]!=best[j] for i,j,_,s,_,_ in relationships))}
    return assignment,report


def build(args):
    source=args.annotations_root.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError(f'Output already exists: {output}; choose a new --output')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    deleted=json.loads((source/'trash.json').read_text(encoding='utf-8'))['samples'] if (source/'trash.json').exists() else {}
    records=[];excluded=[];parents={};hash_owner={}
    def find(key):
        parents.setdefault(key,key)
        if parents[key]!=key:parents[key]=find(parents[key])
        return parents[key]
    for sample in manifest['samples']:
        if sample['id'] in deleted:
            excluded.append(sample['id']);continue
        if sample['scene'] not in SCENES:raise ValueError(f'Unexpected scene: {sample["scene"]}')
        path=source/'annotations'/(sample['id']+'.json')
        annotation=json.loads(path.read_text(encoding='utf-8'))
        if not annotation.get('reviewed'):raise ValueError(f'Unreviewed sample: {sample["id"]}')
        image_path=source/'samples'/sample['id']/'image.png'
        with Image.open(image_path) as opened:image=opened.convert('RGB')
        mask,counts=annotation_mask(annotation,image.size)
        pixel_hash=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
        record={**sample,'annotation':annotation,'mask':mask,'pixels':dict(counts),'image_path':image_path,
                'pixel_sha256':pixel_hash,'annotation_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'sequence':int(re.search(r'\d+',Path(sample['source_name']).stem)[0]),'features':features(image)}
        records.append(record);find(sample['id'])
        for digest in ('source:'+sample['source_sha256'],'pixels:'+pixel_hash):
            if digest in hash_owner:parents[find(sample['id'])]=find(hash_owner[digest])
            else:hash_owner[digest]=sample['id']
    for record in records:record['group_id']=find(record['id'])
    scenes_by_group=defaultdict(set)
    for record in records:scenes_by_group[record['group_id']].add(record['scene'])
    if any(len(scenes)>1 for scenes in scenes_by_group.values()):
        raise ValueError('Identical images occur across scenes; a coupled group split is required')
    rng=random.Random(args.seed);assignments={};scene_reports={}
    for scene in SCENES:
        members=[record for record in records if record['scene']==scene]
        if len(members)<10:raise ValueError(f'Not enough samples in {scene}')
        assignment,report=spread_scene(members,rng,args.attempts)
        assignments.update(assignment);scene_reports[scene]=report
    staging=output.with_name('.'+output.name+'.build_'+uuid.uuid4().hex[:8])
    staging.mkdir(parents=True)
    summaries=[];train_pixels=Counter()
    for record in sorted(records,key=lambda r:r['id']):
        split=assignments[record['id']];sample_id=record['id'];scene=record['scene']
        for base in [staging,staging/'by_scene'/scene]:
            for folder in ('images','masks','annotations'):(base/folder/split).mkdir(parents=True,exist_ok=True)
            shutil.copyfile(record['image_path'],base/'images'/split/(sample_id+'.png'))
            record['mask'].save(base/'masks'/split/(sample_id+'.png'))
            annotation=record['annotation'].copy()
            annotation.update(imagePath=f'../../images/{split}/{sample_id}.png',imageData=None)
            write_json(base/'annotations'/split/(sample_id+'.json'),annotation)
        if split=='train':train_pixels.update(record['pixels'])
        summaries.append({key:record[key] for key in ('id','scene','source_name','source_sha256','pixel_sha256','group_id','bbox','width','height','annotation_sha256')}
                         | {'split':split,'reviewed':True,'pixels':record['pixels']})
    for base in [staging,*[staging/'by_scene'/scene for scene in SCENES]]:
        (base/'splits').mkdir(exist_ok=True)
        for split in SPLITS:
            ids=[r['id'] for r in summaries if r['split']==split and (base==staging or r['scene']==base.name)]
            (base/'splits'/(split+'.txt')).write_text('\n'.join(ids)+'\n',encoding='utf-8')
    report={'version':1,'classes':CLASSES,'line_width':TICK_LINE_WIDTH,'ratio':[7,2,1],'seed':args.seed,
            'policy':'相邻及近似图尽量跨集合分散，完全重复图同组；近似图允许跨集合，评测不代表跨场景泛化。',
            'source':str(source),'excluded_deleted':excluded,'counts':dict(Counter(r['split'] for r in summaries)),
            'scenes':scene_reports,'train_class_pixels':dict(train_pixels),'samples':summaries}
    write_json(staging/'manifest.json',report);write_json(staging/'classes.json',CLASSES)
    write_scene_manifests(staging)
    staging.rename(output)
    print(json.dumps({key:report[key] for key in ('counts','scenes','excluded_deleted')},ensure_ascii=False,indent=2))
    print(f'Dataset: {output}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations-root',type=Path,default=PROJECT.parent/'segmentation_tool/dataset')
    parser.add_argument('--output',type=Path,default=PROJECT/'data/pointer_scale_seg')
    parser.add_argument('--seed',type=int,default=20261113)
    parser.add_argument('--attempts',type=int,default=48)
    args=parser.parse_args()
    if args.attempts<1:parser.error('--attempts must be positive')
    build(args)


if __name__=='__main__':main()
