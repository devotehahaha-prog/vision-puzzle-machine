import unittest
from unittest.mock import Mock, patch
import start_page

class PreviewRefreshTests(unittest.TestCase):
    def test_frames_overwrite_without_clearing(self):
        for paint in (start_page.paint_live_preview, start_page.paint_tune_preview):
            draw, text, frame = Mock(), Mock(), object()
            paint(draw, text, {}, frame)
            paint(draw, text, {}, frame)
            self.assertEqual(draw.image.call_count, 2)
            draw.fill_rect.assert_not_called()
            draw.clear.assert_not_called()
            text.text.assert_not_called()

    def test_failed_transfer_does_not_clear(self):
        for paint in (start_page.paint_live_preview, start_page.paint_tune_preview):
            draw = Mock()
            draw.image.side_effect = OSError("SPI error")
            paint(draw, Mock(), {}, object())
            draw.fill_rect.assert_not_called()
            draw.clear.assert_not_called()

    def test_placeholder_only_before_first_frame(self):
        for paint in (start_page.paint_live_preview, start_page.paint_tune_preview):
            draw = Mock()
            with patch.object(start_page, "centered_text"):
                paint(draw, Mock(), {"card": 0, "muted": 1})
            draw.fill_rect.assert_called_once()
            draw.image.assert_not_called()

    def test_only_active_pixels_are_uploaded_and_borders_once(self):
        from types import SimpleNamespace
        draw = SimpleNamespace(image=Mock(), fill_rect=Mock())
        frame = Mock()
        from layout import LIVE_PREVIEW
        x,y,w,h = LIVE_PREVIEW
        box = (100, 0, w-200, h)
        start_page.paint_live_preview(draw, Mock(), {"scr":0}, frame, box)
        self.assertEqual(draw.fill_rect.call_count, 2)
        frame.crop.assert_called_with((100,0,w-100,h))
        draw.image.assert_called_with(frame.crop.return_value,x+100,y,w-200,h)
        start_page.paint_live_preview(draw, Mock(), {"scr":0}, frame, box)
        self.assertEqual(draw.fill_rect.call_count, 2)
        self.assertEqual(draw.image.call_count, 2)

    def test_invalid_crop_does_not_write_display(self):
        from types import SimpleNamespace
        draw = SimpleNamespace(image=Mock(), fill_rect=Mock())
        start_page.paint_live_preview(draw, Mock(), {}, Mock(), (0,0,0,10))
        draw.image.assert_not_called()
        draw.fill_rect.assert_not_called()
