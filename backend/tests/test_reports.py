from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from backend.app.main import create_app


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.app = create_app()
        self.app.config['REPORTS_DIR'] = self.root
        self.client = self.app.test_client()

    def test_empty_library(self):
        self.assertEqual(self.client.get('/api/reports').json, {'reports': []})

    def test_discovery_lab_assignment_and_pdf_serving(self):
        (self.root / 'lab_01.pdf').write_bytes(b'%PDF-1.4\nfixture')
        (self.root / 'overview.PDF').write_bytes(b'%PDF-1.4\nfixture')
        (self.root / 'README.md').write_text('not a report')
        nested = self.root / 'lab_02'
        nested.mkdir()
        (nested / 'My report.pdf').write_bytes(b'%PDF-1.4\nfixture')
        reports = self.client.get('/api/reports').json['reports']
        self.assertEqual(len(reports), 3)
        by_name = {item['filename']: item for item in reports}
        self.assertEqual(by_name['lab_01.pdf']['lab_number'], 1)
        self.assertEqual(by_name['My report.pdf']['lab_number'], 2)
        self.assertIsNone(by_name['overview.PDF']['lab_number'])
        for report in reports:
            with self.client.get(report['url']) as response:
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.mimetype, 'application/pdf')
                self.assertTrue(response.data.startswith(b'%PDF'))

    def test_refresh_and_restricted_file_access(self):
        self.assertEqual(self.client.get('/api/reports/files/README.md').status_code, 404)
        self.assertEqual(self.client.get('/api/reports/files/../README.md').status_code, 404)
        self.assertEqual(self.client.get('/api/reports/files/missing.pdf').status_code, 404)
        pdf = self.root / 'lab_01.pdf'
        pdf.write_bytes(b'%PDF-1.4')
        self.assertEqual(len(self.client.get('/api/reports').json['reports']), 1)
        pdf.unlink()
        self.assertEqual(self.client.get('/api/reports').json['reports'], [])
