#!/usr/bin/env python3
# Copyright 2025-2026 Lablup Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse, json, os, pathlib, signal, socket, subprocess, time, urllib.error, urllib.request

def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0)); return sock.getsockname()[1]

def call(url, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())

def ready(proc, base):
    for _ in range(600):
        if proc.poll() is not None: raise RuntimeError(f"server exited {proc.returncode}")
        try:
            status, data = call(base+"/v1/models", timeout=2)
            if status == 200: return data["data"][0]["id"]
        except Exception: pass
        time.sleep(.25)
    raise TimeoutError

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--binary", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--arm", required=True); parser.add_argument("--output", required=True)
    args=parser.parse_args(); port=free_port(); base=f"http://127.0.0.1:{port}"
    log_path=pathlib.Path(args.output).with_suffix(".log")
    cmd=[args.binary,"--model",args.model,"--ctx-size","8192","--jinja","--host","127.0.0.1","--port",str(port),"--alias","repro","--parallel","1","--offline"]
    prompts=[
        "Explain in three concise sentences how a solar eclipse occurs and why it is not visible everywhere on Earth.",
        "A warehouse starts with 240 crates. It ships one quarter of them on Monday, receives 36 on Tuesday, then ships one third of the remaining crates on Wednesday. Show the arithmetic and state the final crate count.",
        "Context follows. " + "The blue archive contains cedar records; the red archive contains maple records. "*173 + "Using only the context, state in two complete sentences which archive contains cedar records and which contains maple records.",
    ]
    result={"arm":args.arm,"binary":args.binary,"load_before":os.getloadavg(),"responses":[]}
    with log_path.open("w") as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,text=True,start_new_session=True,env={**os.environ,"RUST_LOG":"info"})
        try:
            model=ready(proc,base)
            for index,prompt in enumerate(prompts):
                status, body=call(base+"/v1/chat/completions", {"model":model,"messages":[{"role":"user","content":prompt}],"max_tokens":128,"temperature":0.0,"stream":False,"chat_template_kwargs":{"enable_thinking":False}})
                result["responses"].append({"index":index,"status":status,"prompt_tokens":(body.get("usage") or {}).get("prompt_tokens"),"completion_tokens":(body.get("usage") or {}).get("completion_tokens"),"choice":(body.get("choices") or [{}])[0]})
                pathlib.Path(args.output).write_text(json.dumps(result,indent=2))
                print(json.dumps(result["responses"][-1]),flush=True)
            result["health"]=call(base+"/health",timeout=10); result["load_after"]=os.getloadavg()
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL); proc.wait(timeout=5)
    result["log"]=str(log_path); pathlib.Path(args.output).write_text(json.dumps(result,indent=2))
if __name__=="__main__": main()
