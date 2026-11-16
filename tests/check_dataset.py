"""Audit the prepared dataset, source review flags, masks and duplicate groups."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

from PIL import Image

from src.data.prepare_dataset import annotation_mask


def main():
    root=Path(__file__).resolve().parents[1]
    dataset=root/'data/pointer_scale_seg';source=root.parent/'segmentation_tool/dataset'
    manifest=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    deleted=json.loads((source/'trash.json').read_text(encoding='utf-8'))['samples']
    splits={name:set((dataset/'splits'/(name+'.txt')).read_text().split()) for name in ('train','val','test')}
    assert not splits['train']&splits['val'] and not splits['train']&splits['test'] and not splits['val']&splits['test']
    assert dict((name,len(ids)) for name,ids in splits.items())=={'train':87,'val':25,'test':12}
    assert set.union(*splits.values())=={r['id'] for r in manifest['samples']}
    groups=defaultdict(set);pixels=Counter()
    for record in manifest['samples']:
        name,split,scene=record['id'],record['split'],record['scene']
        assert name not in deleted and name in splits[split]
        annotation=json.loads((source/'annotations'/(name+'.json')).read_text(encoding='utf-8'))
        assert annotation['reviewed']
        assert hashlib.sha256((source/'annotations'/(name+'.json')).read_bytes()).hexdigest()==record['annotation_sha256']
        image=Image.open(dataset/'images'/split/(name+'.png'));mask=Image.open(dataset/'masks'/split/(name+'.png'))
        expected,_=annotation_mask(annotation,image.size)
        assert image.size==mask.size and mask.mode=='L'
        assert mask.tobytes()==expected.tobytes() and set(mask.tobytes())=={0,1,2}
        for folder in ('images','masks','annotations'):
            extension='.json' if folder=='annotations' else '.png'
            assert (dataset/folder/split/(name+extension)).read_bytes()==(dataset/'by_scene'/scene/folder/split/(name+extension)).read_bytes()
        groups[record['group_id']].add(split);pixels.update(mask.tobytes())
    assert all(len(members)==1 for members in groups.values())
    for scene,report in manifest['scenes'].items():
        actual=Counter(r['split'] for r in manifest['samples'] if r['scene']==scene)
        assert dict(actual)==report['counts']
        individual=json.loads((dataset/'by_scene'/scene/'manifest.json').read_text(encoding='utf-8'))
        assert individual['counts']==report['counts'] and len(individual['samples'])==sum(actual.values())
        assert json.loads((dataset/'by_scene'/scene/'classes.json').read_text())==manifest['classes']
        assert report['similarity_objective_after']<=report['similarity_objective_before']
    print('PASS dataset: 124 reviewed samples, 12 deleted excluded, 87/25/12 splits, exact masks, duplicate groups intact')
    print('Class pixels:',dict(pixels))


if __name__=='__main__':main()
