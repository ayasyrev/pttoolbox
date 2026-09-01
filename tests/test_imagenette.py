import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pttoolbox.data import imagenette


def _response(chunks: list[bytes]) -> MagicMock:
    response = MagicMock()
    response.headers = {"content-length": str(sum(map(len, chunks)))}
    response.iter_content.return_value = chunks
    response.__enter__.return_value = response
    return response


def test_verify_imagenette_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / imagenette.IMAGENETTE_ARCHIVE_FILENAME
    archive_path.write_bytes(b"valid archive")
    expected_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    assert imagenette.verify_imagenette_archive(
        archive_path, expected_sha256=expected_sha256
    )
    assert not imagenette.verify_imagenette_archive(
        archive_path, expected_sha256="0" * 64
    )
    assert not imagenette.verify_imagenette_archive(tmp_path / "missing.tgz")


def test_download_imagenette_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = [b"archive ", b"bytes"]
    response = _response(chunks)
    expected_sha256 = hashlib.sha256(b"".join(chunks)).hexdigest()
    monkeypatch.setattr(imagenette, "IMAGENETTE_ARCHIVE_SHA256", expected_sha256)
    mock_get = MagicMock(return_value=response)
    monkeypatch.setattr(imagenette.requests, "get", mock_get)

    archive_path = imagenette.download_imagenette_archive(tmp_path)

    assert archive_path == tmp_path / imagenette.IMAGENETTE_ARCHIVE_FILENAME
    assert archive_path.read_bytes() == b"".join(chunks)
    assert not archive_path.with_name(f"{archive_path.name}.part").exists()
    mock_get.assert_called_once_with(
        imagenette.IMAGENETTE_ARCHIVE_URL,
        stream=True,
        timeout=30.0,
    )
    response.raise_for_status.assert_called_once_with()


def test_download_reuses_valid_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / imagenette.IMAGENETTE_ARCHIVE_FILENAME
    archive_path.write_bytes(b"existing archive")
    expected_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    monkeypatch.setattr(imagenette, "IMAGENETTE_ARCHIVE_SHA256", expected_sha256)
    mock_get = MagicMock()
    monkeypatch.setattr(imagenette.requests, "get", mock_get)

    result = imagenette.download_imagenette_archive(tmp_path)

    assert result == archive_path
    mock_get.assert_not_called()


def test_download_rejects_invalid_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = _response([b"invalid archive"])
    monkeypatch.setattr(imagenette, "IMAGENETTE_ARCHIVE_SHA256", "0" * 64)
    monkeypatch.setattr(imagenette.requests, "get", MagicMock(return_value=response))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        imagenette.download_imagenette_archive(tmp_path)

    archive_path = tmp_path / imagenette.IMAGENETTE_ARCHIVE_FILENAME
    assert not archive_path.exists()
    assert not archive_path.with_name(f"{archive_path.name}.part").exists()


def test_failed_replacement_preserves_existing_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / imagenette.IMAGENETTE_ARCHIVE_FILENAME
    archive_path.write_bytes(b"old invalid archive")
    response = _response([b"new invalid archive"])
    monkeypatch.setattr(imagenette, "IMAGENETTE_ARCHIVE_SHA256", "0" * 64)
    monkeypatch.setattr(imagenette.requests, "get", MagicMock(return_value=response))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        imagenette.download_imagenette_archive(tmp_path)

    assert archive_path.read_bytes() == b"old invalid archive"
