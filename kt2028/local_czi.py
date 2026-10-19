"""Bounded physical CZI reads using ZEISS pylibCZIrw; never materialize a WSI."""
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image


def native_mpp(xml):
    root=ET.fromstring(xml)
    result=[]
    for axis in ('X','Y'):
        nodes=root.findall(f'.//Scaling/Items/Distance[@Id="{axis}"]/Value')
        values={float(x.text)*1e6 for x in nodes}
        if len(values)!=1 or not all(math.isfinite(x) and x>0 for x in values):
            raise ValueError('Missing or ambiguous CZI physical scaling')
        result.append(values.pop())
    return tuple(result)


class LocalCZI:
    def __init__(self,path,maximum_upsampling_fraction=.02):
        from pylibCZIrw import czi
        self.path=Path(path)
        self.context=czi.open_czi(str(self.path),cache_options=czi.CacheOptions(max_memory_usage=512*1024**2))
        self.doc=self.context.__enter__()
        try:
            self.mpp=native_mpp(self.doc.raw_metadata)
            scenes=self.doc.scenes_bounding_rectangle
            if len(scenes)>1:
                raise ValueError('Multiple CZI scenes require explicit mapping; do not merge blindly')
            self.scene=next(iter(scenes)) if scenes else None
            self.bounds=tuple(scenes[self.scene] if scenes else self.doc.total_bounding_rectangle)
            self.origin=self.bounds[:2]; self.wh=self.bounds[2:]
            for key,(a,b) in self.doc.total_bounding_box.items():
                if key not in ('X','Y') and b-a!=1:
                    raise ValueError('Non-singleton CZI plane requires explicit selection')
            if any(p.lower()!='bgr24' for p in self.doc.pixel_types.values()):
                raise ValueError('Only verified brightfield BGR24 is supported')
            self.plane={k:v[0] for k,v in self.doc.total_bounding_box.items() if k in ('T','Z','C')}
            self.image_wh_um=tuple(a*b for a,b in zip(self.wh,self.mpp))
            self.maximum_upsampling_fraction=maximum_upsampling_fraction
        except Exception:
            self.context.__exit__(None,None,None)
            raise

    def close(self): self.context.__exit__(None,None,None)
    def __enter__(self): return self
    def __exit__(self,*args): self.close()

    def read_rgb(self,roi,downsample,*,maximum_pixels=4_300_000):
        if maximum_pixels>20_000_000 or math.ceil(roi[2]/downsample)*math.ceil(roi[3]/downsample)>maximum_pixels:
            raise ValueError('CZI output allocation exceeds bound')
        value=self.doc.read(roi=tuple(int(x) for x in roi),plane=self.plane,scene=self.scene,
                            zoom=1/downsample,pixel_type='Bgr24',background_pixel=(1.,1.,1.))
        value=np.asarray(value)
        if value.ndim!=3 or value.shape[-1]!=3 or value.dtype!=np.uint8:
            raise ValueError('Unexpected CZI decoded pixel shape/type')
        if value.shape[:2]!=(roi[3]//downsample,roi[2]//downsample):
            raise ValueError('CZI scaling output shape differs from aligned ROI')
        return Image.fromarray(value[:,:,::-1].copy())

    def read_patch(self,x_um,y_um,target_mpp,pixels=224,*,native_reference=False):
        if min(x_um,y_um)<0 or not all(math.isfinite(v) for v in (x_um,y_um,target_mpp)) or target_mpp<=0:
            raise ValueError('Invalid physical patch')
        if max(self.mpp)>target_mpp*(1+self.maximum_upsampling_fraction):
            raise ValueError('Native CZI resolution insufficient')
        d=1 if native_reference else 2**max(0,math.floor(math.log2(target_mpp/max(self.mpp))))
        xy=np.asarray(self.origin)+np.asarray([x_um,y_um])/self.mpp
        left=np.floor(xy/d).astype(np.int64)*d
        extent=pixels*target_mpp/np.asarray(self.mpp)
        size=np.ceil((xy-left+extent)/d).astype(np.int64)*d
        # Native references are restricted to small ROIs; not a full-WSI fallback.
        image=self.read_rgb((*left,*size),d,maximum_pixels=20_000_000 if native_reference else 4_300_000)
        offset=(xy-left)/d
        return image.transform((pixels,pixels),Image.Transform.EXTENT,
            (*offset,*(offset+extent/d)),resample=Image.Resampling.BILINEAR,fillcolor='white')

    def thumbnail(self,maximum_edge):
        d=2**max(0,math.ceil(math.log2(max(self.wh)/maximum_edge)))
        left=np.floor(np.asarray(self.origin)/d).astype(np.int64)*d
        size=np.ceil((np.asarray(self.origin)-left+self.wh)/d).astype(np.int64)*d
        # Align read to document coordinates, then crop to the local-image origin.
        image=self.read_rgb((*left,*size),d)
        offset=(np.asarray(self.origin)-left)/d
        shape=(max(1,round(self.wh[0]/d)),max(1,round(self.wh[1]/d)))
        return image.transform(shape,Image.Transform.EXTENT,
            (*offset,*(offset+np.asarray(self.wh)/d)),resample=Image.Resampling.BILINEAR,fillcolor='white')

    def metadata(self):
        import importlib.metadata
        return dict(dimensions=list(self.wh),native_mpp_xy=list(self.mpp),
            image_wh_um=list(self.image_wh_um),document_origin_level0_xy=list(self.origin),
            selected_scene=self.scene,pixel_type='Bgr24',color_policy='explicit BGR to RGB; no stain normalization/ICC transform',
            native_metadata_sha256=hashlib.sha256(self.doc.raw_metadata.encode()).hexdigest(),
            reader='ZEISS pylibCZIrw',reader_version=importlib.metadata.version('pylibCZIrw'),
            sampling='power-of-two zoom at least as resolved as target; aligned document ROI; physical bilinear EXTENT',
            coordinate_definition='local level0 top-left; add document_origin_level0_xy for CZI document coordinates',
            maximum_native_upsampling_fraction=self.maximum_upsampling_fraction)
