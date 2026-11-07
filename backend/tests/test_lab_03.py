import base64
import io
import json
import unittest

import cv2
import numpy as np

from backend.app.main import create_app


class LabThreeTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()
        self.image = np.tile(np.arange(80, 144, dtype=np.uint8), (64, 1))

    def post(self, image=None, options=None, contents=None):
        if contents is None:
            contents = cv2.imencode('.png', self.image if image is None else image)[1].tobytes()
        return self.client.post('/api/labs/3/process', data={
            'image': (io.BytesIO(contents), 'input.png'), 'options': json.dumps(options or {})})

    def decode(self, output):
        raw = base64.b64decode(output['url'].split(',', 1)[1])
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)

    def test_grayscale_methods_and_histogram_counts(self):
        response = self.post()
        self.assertEqual(response.status_code, 200)
        result = response.json
        expected = {
            'original': self.image,
            'equalized': cv2.equalizeHist(self.image),
            'clahe': cv2.createCLAHE(2.0, (8, 8)).apply(self.image),
            'stretched': np.rint((self.image.astype(float) - 80) * 255 / 63).astype(np.uint8),
        }
        for key, image in expected.items():
            with self.subTest(method=key):
                output = result['images'][key]
                np.testing.assert_array_equal(self.decode(output), image)
                self.assertEqual(sum(output['histogram']), image.size)
                np.testing.assert_array_equal(output['histogram'], np.bincount(image.ravel(), minlength=256))
                plot = base64.b64decode(output['histogram_url'].split(',', 1)[1])
                self.assertTrue(plot.startswith(b'\x89PNG\r\n\x1a\n'))
        self.assertEqual(result['images']['stretched']['statistics']['min'], 0)
        self.assertEqual(result['images']['stretched']['statistics']['max'], 255)

    def test_color_changes_only_y_before_reconstruction(self):
        color = cv2.merge([self.image, np.full_like(self.image, 90), np.full_like(self.image, 110)])
        response = self.post(color, {'mode': 'color', 'clip_limit': 3, 'tile_grid': 4})
        self.assertEqual(response.status_code, 200)
        result = response.json
        yuv = cv2.cvtColor(color, cv2.COLOR_BGR2YUV)
        np.testing.assert_array_equal(self.decode(result['images']['original']), color)
        for name, plane in [('equalized', cv2.equalizeHist(yuv[:, :, 0])),
                            ('clahe', cv2.createCLAHE(3, (4, 4)).apply(yuv[:, :, 0]))]:
            expected = yuv.copy()
            expected[:, :, 0] = plane
            np.testing.assert_array_equal(self.decode(result['images'][name]), cv2.cvtColor(expected, cv2.COLOR_YUV2BGR))
            np.testing.assert_array_equal(result['images'][name]['histogram'], np.bincount(plane.ravel(), minlength=256))

    def test_constant_and_single_pixel(self):
        for shape in [(8, 8), (1, 1)]:
            image = np.full(shape, 100, np.uint8)
            response = self.post(image)
            self.assertEqual(response.status_code, 200)
            result = response.json
            np.testing.assert_array_equal(self.decode(result['images']['stretched']), image)
            np.testing.assert_array_equal(self.decode(result['images']['equalized']), image)

    def test_bad_requests(self):
        self.assertEqual(self.client.post('/api/labs/3/process').status_code, 400)
        for contents in [b'', b'not an image']:
            self.assertEqual(self.post(contents=contents).status_code, 400)
        for options in [{'mode': 'rgb'}, {'mode': []}, {'clip_limit': 0}, {'clip_limit': True},
                        {'clip_limit': '2'}, {'clip_limit': float('nan')},
                        {'tile_grid': 0}, {'tile_grid': 33}, {'tile_grid': 2.5}]:
            with self.subTest(options=options):
                self.assertEqual(self.post(options=options).status_code, 400)

    def test_script_available(self):
        with self.client.get('/assets/lab_03.js') as response:
            self.assertEqual(response.status_code, 200)
