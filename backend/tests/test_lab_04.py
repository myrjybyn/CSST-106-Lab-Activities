import base64
import io
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from backend.app.main import create_app
from backend.app.labs import lab_04


class LabFourTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()
        self.image = np.random.default_rng(42).integers(0, 256, (240, 240), dtype=np.uint8)
        self.contents = cv2.imencode('.png', self.image)[1].tobytes()

    def post(self, contents=None, options=None):
        contents = self.contents if contents is None else contents
        return self.client.post('/api/labs/4/process', data={
            'image_a': (io.BytesIO(contents), 'a.png'),
            'image_b': (io.BytesIO(contents), 'b.png'), 'options': json.dumps(options or {})})

    def test_matching_and_descriptor_information(self):
        response = self.post()
        self.assertEqual(response.status_code, 200)
        algorithms = {item['name']: item for item in response.json['algorithms']}
        for name, columns, dtype in [('SIFT', 128, 'float32'), ('ORB', 32, 'uint8')]:
            result = algorithms[name]
            self.assertTrue(result['available'])
            self.assertGreater(result['matches'], 0)
            self.assertEqual(result['descriptor_a'][1], columns)
            self.assertEqual(result['descriptor_dtype'], dtype)
            self.assertLessEqual(result['used_a'], 1000)
            self.assertLessEqual(result['displayed_matches'], 50)
            self.assertGreaterEqual(result['detect_ms_a'], 0)
            for image in result['images'].values():
                raw = base64.b64decode(image.split(',', 1)[1])
                self.assertIsNotNone(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR))

    def test_unavailable_surf_does_not_produce_alternative_output(self):
        original = lab_04.make_detector
        def without_surf(name, threshold=400):
            if name in ('SURF', 'U-SURF'):
                raise AttributeError('unavailable')
            return original(name, threshold)
        with patch.object(lab_04, 'make_detector', side_effect=without_surf):
            result = self.post().json
        for algorithm in result['algorithms'][:2]:
            self.assertFalse(algorithm['available'])
            self.assertNotIn('images', algorithm)
            self.assertIn('OPENCV_ENABLE_NONFREE', algorithm['reason'])

    def test_surf_and_upright_configuration(self):
        # Constructor wiring only; real SURF execution requires a nonfree build.
        with patch.object(cv2.xfeatures2d, 'SURF_create') as constructor:
            lab_04.make_detector('SURF', 700)
            constructor.assert_called_with(hessianThreshold=700, upright=False)
            lab_04.make_detector('U-SURF', 700)
            constructor.assert_called_with(hessianThreshold=700, upright=True)

    def test_textureless_images(self):
        contents = cv2.imencode('.png', np.zeros((80, 80), np.uint8))[1].tobytes()
        response = self.post(contents)
        self.assertEqual(response.status_code, 200)
        for item in response.json['algorithms']:
            if item['available']:
                self.assertEqual(item['matches'], 0)
                self.assertIsNone(item['descriptor_a'])

    def test_invalid_inputs(self):
        self.assertEqual(self.client.post('/api/labs/4/process').status_code, 400)
        self.assertEqual(self.post(b'').status_code, 400)
        self.assertEqual(self.post(b'not an image').status_code, 400)
        for options in [{'ratio': 0}, {'ratio': 1}, {'ratio': True}, {'ratio': float('nan')},
                        {'hessian_threshold': 0}, {'hessian_threshold': 2.5}]:
            with self.subTest(options=options):
                self.assertEqual(self.post(options=options).status_code, 400)

    def test_capabilities_and_script(self):
        response = self.client.get('/api/labs/4/capabilities')
        self.assertEqual(response.status_code, 200)
        self.assertEqual([item['name'] for item in response.json['algorithms']], ['SURF', 'U-SURF', 'SIFT', 'ORB'])
        with self.client.get('/assets/lab_04.js') as response:
            self.assertEqual(response.status_code, 200)
