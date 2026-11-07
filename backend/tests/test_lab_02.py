import base64
import io
import json
import unittest

import cv2
import numpy as np

from backend.app.main import create_app


class LabTwoTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()
        self.image = np.zeros((60, 80), dtype=np.uint8)
        self.image[15:45, 20:60] = 255
        self.contents = cv2.imencode('.png', self.image)[1].tobytes()

    def post(self, options=None, contents=None):
        return self.client.post('/api/labs/2/process', data={
            'image': (io.BytesIO(self.contents if contents is None else contents), 'input.png'),
            'options': json.dumps(options or {}),
        })

    def decode(self, result, name):
        raw = base64.b64decode(result['images'][name]['url'].split(',')[1])
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)

    def test_defaults_match_handout(self):
        response = self.post()
        self.assertEqual(response.status_code, 200)
        result = response.json
        blurred = cv2.GaussianBlur(self.image, (3, 3), 0)
        x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(x, y)
        expected = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        np.testing.assert_array_equal(self.decode(result, 'original'), self.image)
        np.testing.assert_array_equal(self.decode(result, 'blurred'), blurred)
        np.testing.assert_array_equal(self.decode(result, 'sobel'), expected)
        np.testing.assert_array_equal(self.decode(result, 'canny'), cv2.Canny(blurred, 100, 200))
        self.assertGreater(result['properties']['canny_edge_pixels'], 0)
        self.assertEqual(result['parameters']['lower_threshold'], 100)

    def test_custom_thresholds(self):
        result = self.post({'lower_threshold': 25, 'upper_threshold': 75}).json
        expected = cv2.Canny(cv2.GaussianBlur(self.image, (3, 3), 0), 25, 75)
        np.testing.assert_array_equal(self.decode(result, 'canny'), expected)

    def test_blank_image_has_no_edges(self):
        for shape in [(20, 20), (1, 1)]:
            with self.subTest(shape=shape):
                contents = cv2.imencode('.png', np.zeros(shape, dtype=np.uint8))[1].tobytes()
                response = self.post(contents=contents)
                self.assertEqual(response.status_code, 200)
                result = response.json
                self.assertEqual(result['properties']['canny_edge_pixels'], 0)
                self.assertFalse(self.decode(result, 'sobel').any())

    def test_bad_requests(self):
        self.assertEqual(self.client.post('/api/labs/2/process').status_code, 400)
        self.assertEqual(self.post(contents=b'bad image').status_code, 400)
        self.assertEqual(self.post(contents=b'').status_code, 400)
        for options in [{'lower_threshold': 201}, {'lower_threshold': -1},
                        {'upper_threshold': 256}, {'lower_threshold': 100.5},
                        {'lower_threshold': True}, {'upper_threshold': 100}]:
            with self.subTest(options=options):
                self.assertEqual(self.post(options).status_code, 400)

    def test_lab_registry(self):
        self.assertEqual([lab['id'] for lab in self.client.get('/api/labs').json['labs']], ['lab-01', 'lab-02', 'lab-03', 'lab-04'])
        with self.client.get('/assets/lab_02.js') as response:
            self.assertEqual(response.status_code, 200)
