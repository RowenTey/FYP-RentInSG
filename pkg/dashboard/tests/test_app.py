import pytest
from streamlit.testing.v1 import AppTest


def test_streamlit_app():
    at = AppTest.from_file("app.py")
    at.run()
    assert not at.exception


if __name__ == "__main__":
    pytest.main()
