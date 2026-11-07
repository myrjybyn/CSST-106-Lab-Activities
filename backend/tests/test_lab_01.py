import base64
import unittest

import cv2
import numpy as np

from backend.app.main import create_app


class LabOneTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()
        self.image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        self.encoded = cv2.imencode('.png', self.image)[1].tobytes()

    def post(self, contents=None, options='{}'):
        import io
        return self.client.post('/api/labs/1/process', data={
            'image': (io.BytesIO(self.encoded if contents is None else contents), 'test.png'),
            'options': options,
        })

    def decode(self, result, key):
        raw = base64.b64decode(result['images'][key]['url'].split(',', 1)[1])
        return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    def test_web_and_registry(self):
        for path in ['/', '/assets/app.js', '/assets/styles.css']:
            with self.client.get(path) as response:
                self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get('/api/labs').json['labs'][0]['number'], 1)

    def test_outputs_match_original_pixels_and_operations(self):
        response = self.post(options='{"pixel_x":2,"pixel_y":1,"blue":255,"green":0,"red":0,"crop_x":1,"crop_y":2,"crop_width":3,"crop_height":2,"resize_width":4,"resize_height":3}')
        self.assertEqual(response.status_code, 200)
        data = response.json
        self.assertEqual(data['properties']['shape'], [6, 8, 3])
        self.assertEqual(data['pixel']['before_bgr'], self.image[1, 2].tolist())
        np.testing.assert_array_equal(self.decode(data, 'original'), self.image)
        expected = self.image.copy()
        expected[1, 2] = [255, 0, 0]
        np.testing.assert_array_equal(self.decode(data, 'pixel_modified'), expected)
        np.testing.assert_array_equal(self.decode(data, 'cropped'), self.image[2:4, 1:4])
        np.testing.assert_array_equal(self.decode(data, 'grayscale'), cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY))
        np.testing.assert_array_equal(self.decode(data, 'rotated'), np.rot90(self.image, k=-1))
        self.assertEqual(self.decode(data, 'resized').shape, (3, 4, 3))

    def test_bad_inputs(self):
        self.assertEqual(self.client.post('/api/labs/1/process').status_code, 400)
        self.assertEqual(self.post(b'not an image').status_code, 400)
        self.assertEqual(self.post(b'').status_code, 400)
        for options in ['{', '[]', '{"pixel_x":8}', '{"crop_x":7,"crop_width":2}', '{"resize_width":0}', '{"red":256}', '{"pixel_x":1.5}', '{"pixel_x":true}']:
            with self.subTest(options=options):
                self.assertEqual(self.post(options=options).status_code, 400)

    def test_grayscale_and_single_pixel_images(self):
        for image in [np.zeros((4, 4), dtype=np.uint8), np.zeros((1, 1, 3), dtype=np.uint8)]:
            with self.subTest(shape=image.shape):
                contents = cv2.imencode('.png', image)[1].tobytes()
                self.assertEqual(self.post(contents).status_code, 200)

    def test_upload_limit(self):
        app = create_app()
        app.config['MAX_CONTENT_LENGTH'] = 100
        response = app.test_client().post('/api/labs/1/process', data=b'x' * 101, content_type='application/octet-stream')
        self.assertEqual(response.status_code, 413)
        self.assertIn('error', response.json)


if __name__ == '__main__':
    unittest.main()
