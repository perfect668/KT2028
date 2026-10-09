import pytest
from kt2028.remote_rgb import CachedRangeFile


def test_offline_cache_miss_never_falls_back_to_network(tmp_path,monkeypatch):
    r=CachedRangeFile('https://ftp.ebi.ac.uk/test',11,tmp_path,22,cache_only=True)
    def forbidden(*args,**kwargs):raise AssertionError('Unexpected network access')
    monkeypatch.setattr(r.session,'get',forbidden)
    with pytest.raises(FileNotFoundError):r.read(3)
    (r.cache/'0_11.bin').write_bytes(b'hello-world')
    assert r.read(5)==b'hello'
    r.seek(-5,2)
    assert r.read()==b'world'
    r.close()


def test_truncated_cached_fragment_rejected(tmp_path):
    r=CachedRangeFile('https://ftp.ebi.ac.uk/test',11,tmp_path,22,cache_only=True)
    (r.cache/'0_11.bin').write_bytes(b'bad')
    with pytest.raises(ValueError):r.read(3)
    r.close()
