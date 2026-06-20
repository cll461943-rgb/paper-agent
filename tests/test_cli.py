import sys
from unittest.mock import patch
import pytest

from scholar_agent.cli import main


def test_cli_help():
    # 测试 -h 帮助输出，会抛出 SystemExit
    with patch.object(sys, "argv", ["scholar-agent", "-h"]):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0


def test_cli_run_mock(tmp_path):
    # 测试 mock 模式下的搜索过程，并将结果输出到临时文件
    output_file = tmp_path / "result.json"
    
    with patch.object(
        sys,
        "argv",
        [
            "scholar-agent",
            "MIMIC-CXR Radiology Report Generation multimodal LLM",
            "--mode",
            "mock",
            "--output",
            str(output_file),
        ],
    ):
        with patch("builtins.print") as mock_print:
            # 运行命令，不抛出 SystemExit 说明正常退出 (或者可能没有 sys.exit 而是正常退出)
            main()
            
            # 确认生成了 output_file 且内容非空
            assert output_file.exists()
            content = output_file.read_text(encoding="utf-8")
            assert "highly_relevant_papers" in content
            assert "run_metrics" in content
