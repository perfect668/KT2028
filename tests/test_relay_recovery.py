import subprocess
import urllib.error
from kt2028 import relay_cohort
from kt2028.prefetch_rgb_cache import RelayFile


def test_temporary_ssh_failure_is_retried_without_skipping_asset(monkeypatch):
    calls=[]
    def run(*args,**kwargs):
        calls.append(args)
        if len(calls)==1:raise subprocess.TimeoutExpired(args[0],60)
        return subprocess.CompletedProcess(args[0],0)
    monkeypatch.setattr(relay_cohort.subprocess,'run',run)
    monkeypatch.setattr(relay_cohort.time,'sleep',lambda _:None)
    assert relay_cohort.transport(['ssh','example'],60).returncode==0
    assert len(calls)==2


def test_completed_fragment_survives_a_network_retry_and_is_reused(tmp_path,monkeypatch):
    from kt2028 import prefetch_rgb_cache as module
    calls=[]
    class Response:
        status=206
        headers={'Content-Range':'bytes 0-10/11'}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,n):return b'hello-world'
    def urlopen(*args,**kwargs):
        calls.append(args)
        if len(calls)==1:raise urllib.error.URLError('temporary connection reset')
        return Response()
    monkeypatch.setattr(module.urllib.request,'urlopen',urlopen)
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    r=RelayFile({'path':'test.zip','size':11},tmp_path/'ranges',100,1)
    assert r.chunk(0,11)==b'hello-world'
    assert r.chunk(0,11)==b'hello-world'
    assert len(calls)==2 and r.bytes==11


def test_short_successful_http_body_is_retried_and_charged(tmp_path,monkeypatch):
    from kt2028 import prefetch_rgb_cache as module
    calls=[]
    class Response:
        status=206
        headers={'Content-Range':'bytes 0-10/11'}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,n):return b'hello' if len(calls)==1 else b'hello-world'
    def urlopen(*args,**kwargs):
        calls.append(args);return Response()
    monkeypatch.setattr(module.urllib.request,'urlopen',urlopen)
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    r=RelayFile({'path':'test.zip','size':11},tmp_path/'ranges',100,1)
    assert r.chunk(0,11)==b'hello-world'
    assert len(calls)==2 and r.bytes==16
