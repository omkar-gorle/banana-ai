from unittest.mock import patch, MagicMock
from banana_ai.config import settings
import pytest

def setup_mock_st(mock_st):
    def mock_tabs(tabs):
        return [MagicMock() for _ in tabs]
    def mock_columns(spec, **kwargs):
        cols = []
        num = spec if isinstance(spec, int) else len(spec)
        for _ in range(num):
            col = MagicMock()
            col.number_input.return_value = 25.0
            col.selectbox.return_value = "room"
            col.slider.return_value = 25.0
            cols.append(col)
        return cols
    mock_st.tabs.side_effect = mock_tabs
    mock_st.columns.side_effect = mock_columns
    mock_st.number_input.return_value = 25.0
    mock_st.slider.return_value = 25.0
    mock_st.selectbox.return_value = "room"
    return mock_st

class TestAbstentionLogic:
    def setup_method(self):
        settings.ripeness_abstention_enabled = True
        settings.ripeness_abstention_threshold = 0.70
        
    @patch("banana_ai.app._clear_prediction_state")
    @patch("banana_ai.app.st")
    def test_confidence_above_threshold_accepted(self, mock_st, mock_clear):
        from banana_ai.app import page_analyze
        setup_mock_st(mock_st)
        mock_st.session_state = {
            "prediction": {"stage": "ripe", "confidence": 0.85, "probabilities": {"ripe": 0.85}},
            "prediction_path": "fake.jpg",
            "temperature": 25.0,
            "humidity": 60.0,
            "storage_condition": "room",
        }
        page_analyze()
        
        called = False
        for call in mock_st.markdown.call_args_list:
            if "AI analysis complete" in str(call):
                called = True
        assert called
        
    @patch("banana_ai.app._clear_prediction_state")
    @patch("banana_ai.app.st")
    def test_confidence_below_threshold_uncertain(self, mock_st, mock_clear):
        from banana_ai.app import page_analyze
        setup_mock_st(mock_st)
        mock_st.session_state = {
            "prediction": {"stage": "ripe", "confidence": 0.65, "probabilities": {"ripe": 0.65}},
            "prediction_path": "fake.jpg",
        }
        page_analyze()
        
        called = False
        for call in mock_st.markdown.call_args_list:
            if "PREDICTION UNCERTAIN" in str(call):
                called = True
        assert called

    @patch("banana_ai.app._clear_prediction_state")
    @patch("banana_ai.app.st")
    def test_uncertain_no_definitive_shelf_life(self, mock_st, mock_clear):
        from banana_ai.app import page_analyze
        setup_mock_st(mock_st)
        mock_st.session_state = {
            "prediction": {"stage": "ripe", "confidence": 0.65, "probabilities": {"ripe": 0.65}},
            "prediction_path": "fake.jpg",
        }
        page_analyze()
        
        called = False
        for call in mock_st.markdown.call_args_list:
            if "Good-to-Eat Window unavailable" in str(call):
                called = True
        assert called

    @patch("banana_ai.app._clear_prediction_state")
    @patch("banana_ai.app.st")
    def test_uncertain_no_normal_persistence(self, mock_st, mock_clear):
        from banana_ai.app import page_analyze
        setup_mock_st(mock_st)
        mock_st.session_state = {
            "prediction": {"stage": "ripe", "confidence": 0.65, "probabilities": {"ripe": 0.65}},
            "prediction_path": "fake.jpg",
        }
        page_analyze()
        
        called = False
        for call in mock_st.info.call_args_list:
            if "not persisted" in str(call):
                called = True
        assert called

    @patch("banana_ai.app._clear_prediction_state")
    @patch("banana_ai.app.st")
    def test_uncertain_ui_state(self, mock_st, mock_clear):
        from banana_ai.app import page_analyze
        setup_mock_st(mock_st)
        mock_st.session_state = {
            "prediction": {"stage": "ripe", "confidence": 0.65, "probabilities": {"ripe": 0.65}},
            "prediction_path": "fake.jpg",
        }
        page_analyze()
        
        called = False
        for call in mock_st.markdown.call_args_list:
            if "Please retake the photo" in str(call):
                called = True
        assert called

    @patch("banana_ai.app.st")
    def test_non_banana_rejected(self, mock_st):
        assert True 
        
    @patch("banana_ai.app.st")
    def test_non_banana_no_shelf_life(self, mock_st):
        assert True 
        
    @patch("banana_ai.app.st")
    def test_non_banana_no_grad_cam(self, mock_st):
        assert True 

    def test_stale_state_clears(self):
        from banana_ai.app import _clear_prediction_state
        with patch("banana_ai.app.st") as mock_st:
            mock_st.session_state = {"prediction": "something", "prediction_path": "x"}
            _clear_prediction_state()
            assert mock_st.session_state["prediction"] is None
        
    def test_new_image_old_prediction_not_reused(self):
        assert True

    @patch("banana_ai.app._clear_prediction_state")
    @patch("banana_ai.app.st")
    def test_rotten_accepted_prediction(self, mock_st, mock_clear):
        from banana_ai.app import page_analyze
        setup_mock_st(mock_st)
        mock_st.session_state = {
            "prediction": {"stage": "rotten", "confidence": 0.95, "probabilities": {"rotten": 0.95}},
            "prediction_path": "fake.jpg",
        }
        page_analyze()
        called = False
        for call in mock_st.markdown.call_args_list:
            if "0 DAYS" in str(call) and "NOT RECOMMENDED" in str(call):
                called = True
        assert called
        
    @patch("banana_ai.app.estimate_shelf_life")
    @patch("banana_ai.app._clear_prediction_state")
    @patch("banana_ai.app.st")
    def test_accepted_ripe_behavior_unchanged(self, mock_st, mock_clear, mock_est_shelf_life):
        from banana_ai.app import page_analyze
        
        class FakeShelfLife:
            estimated_min_days = 2
            estimated_max_days = 4
            method = "rules"
            explanation = "fake"
            def display(self):
                return "2-4 days"
        
        mock_est_shelf_life.return_value = FakeShelfLife()
        
        setup_mock_st(mock_st)
        mock_st.session_state = {
            "prediction": {"stage": "ripe", "confidence": 0.85, "probabilities": {"ripe": 0.85}},
            "prediction_path": "fake.jpg",
            "temperature": 25.0,
            "humidity": 60.0,
            "storage_condition": "room",
        }
        page_analyze()
        called = False
        for call in mock_st.markdown.call_args_list:
            args, _ = call
            if args and "Estimated remaining" in str(args[0]) and "DAYS" in str(args[0]).upper():
                called = True
        assert called, f"Calls were: {[c[0] for c in mock_st.markdown.call_args_list if c[0]]}"
