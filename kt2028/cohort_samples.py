"""Build unlabelled acquisition inputs from the verified official split manifests."""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--s1',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    images=json.loads((args.s1/'ome_file_list.json').read_text())
    image_map={Path(i['path']).name.removesuffix('.ome.zarr.zip'):i for i in images if '/SR386_' in i['path']}
    rows=[];cases=set();slides=set()
    for split in ['train','validate','test']:
        manifest=json.loads((args.s1/(split+'_asset_manifest.json')).read_text())
        for case in manifest['patients']:
            key=case['canonical_case_key']
            if key in cases:raise ValueError('Cross-split case duplication')
            cases.add(key)
            for s in case['slides']:
                slide=s['slide_id']
                if slide in slides:raise ValueError('Cross-case slide duplication')
                if slide not in image_map:raise ValueError('Missing official image')
                slides.add(slide)
                rows.append({'slide_id':slide,'source':image_map[slide],'fine_path':s['features_path']})
    with args.output.open('x') as f:json.dump(rows,f,indent=2)
    print('COHORT_INPUTS '+json.dumps({'cases':len(cases),'slides':len(slides),'labels_in_inputs':False}),flush=True)

if __name__=='__main__':main()
