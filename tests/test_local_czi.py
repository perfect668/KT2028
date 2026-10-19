"""Reader software fixtures; actual CZI/UNI acceptance is recorded separately."""
import unittest
import numpy as np
from kt2028.local_czi import LocalCZI,native_mpp


class CziReaderTests(unittest.TestCase):
    def test_native_scaling_uses_si_value_not_display_unit(self):
        xml='<Metadata><Scaling><Items><Distance Id="X"><Value>1e-7</Value><DefaultUnitFormat>um</DefaultUnitFormat></Distance><Distance Id="Y"><Value>2e-7</Value></Distance></Items></Scaling></Metadata>'
        np.testing.assert_allclose(native_mpp(xml),(.1,.2),rtol=0,atol=1e-15)

    def test_missing_scaling_is_not_guessed(self):
        with self.assertRaises(ValueError): native_mpp('<Metadata/>')

    def test_bgr_conversion_and_white_background(self):
        class Doc:
            def read(self,**kwargs):
                self.args=kwargs
                return np.tile(np.array([11,22,33],dtype=np.uint8),(2,3,1))
        reader=LocalCZI.__new__(LocalCZI); reader.doc=Doc(); reader.plane={}; reader.scene=0
        rgb=reader.read_rgb((0,0,6,4),2)
        self.assertEqual(rgb.getpixel((0,0)),(33,22,11))
        self.assertEqual(reader.doc.args['background_pixel'],(1.,1.,1.))

    def test_full_wsi_allocation_is_rejected_before_decoding(self):
        reader=LocalCZI.__new__(LocalCZI)
        with self.assertRaisesRegex(ValueError,'allocation'):
            reader.read_rgb((0,0,200000,200000),1)


if __name__=='__main__':unittest.main()
