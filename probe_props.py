import time, requests, os
os.environ['FAST_PROPS_TEST']='1'
os.environ['PROPS_VERBOSE']='1'
url='http://127.0.0.1:8080/props'
print('Probing', url)
for i in range(3):
    t=time.perf_counter(); r=requests.get(url); dt=time.perf_counter()-t
    print(f'Attempt {i+1}: status={r.status_code} bytes={len(r.content)} ms={dt*1000:.1f} fast={r.headers.get("X-Fast")}')
