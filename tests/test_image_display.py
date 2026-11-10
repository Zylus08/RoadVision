import base64
from io import BytesIO
import unittest

import numpy as np
from PIL import Image

from src.visualization import image_to_data_uri


class TestImageDisplay(unittest.TestCase):
    def test_image_to_data_uri_preserves_rgb_pixels(self):
        image = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
        uri = image_to_data_uri(Image.fromarray(image))

        self.assertTrue(uri.startswith("data:image/png;base64,"))
        encoded_image = base64.b64decode(uri.partition(",")[2])
        with Image.open(BytesIO(encoded_image)) as decoded:
            self.assertEqual(decoded.mode, "RGB")
            self.assertEqual(decoded.size, (2, 1))
            np.testing.assert_array_equal(np.array(decoded), image)


if __name__ == "__main__":
    unittest.main()
