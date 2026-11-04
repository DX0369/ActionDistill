"""Exact task identity, union-find and deterministic group holdouts."""
import re
from .common import digest


def source_task_id(source, rule):
    """Parse the canonical JSON-record source locator using an explicit path rule."""
    if not isinstance(source,str):raise ValueError('source must be a JSON-record locator')
    path,sep,index=source.rpartition(':')
    if not sep or index!='1':raise ValueError(f'expected single JSON-record source ending :1: {source}')
    match=re.fullmatch(rule['path_pattern'],path)
    if match is None or match['competition'] in {'.','..'}:
        raise ValueError(f'source does not match competition path rule: {source}')
    return rule['prefix']+match['competition']


def swe_task_id(raw, repo=None):
    if not isinstance(raw,str) or not raw:
        raise ValueError('missing SWE canonical task id')
    if repo is not None:
        if not isinstance(repo,str) or len(repo.split('/'))!=2 or any(not p for p in repo.split('/')):
            raise ValueError('SWE repo must be owner/repository')
        owner,name = repo.split('/')
        prefixes = (f'{owner}_{name}_pr',f'{owner}__{name}-')
        matches=[raw[len(p):] for p in prefixes if raw.lower().startswith(p.lower())]
        if len(matches)!=1 or not re.fullmatch(r'[1-9][0-9]*',matches[0]):
            raise ValueError(f'SWE id/repo mismatch or unknown scheme: {raw}, {repo}')
        number = matches[0]
    else:
        match = re.fullmatch(r'([A-Za-z0-9-]+)__(.+)-([1-9][0-9]*)',raw)
        if match is None:
            raise ValueError(f'unknown SWE-rebench PR id: {raw}')
        owner,name,number = match.groups()
    return f'swe_pr:{owner.lower()}__{name.lower()}-{number}'


class UnionFind:
    def __init__(self, keys):
        self.parent={k:k for k in keys}
    def find(self,k):
        while self.parent[k]!=k:
            self.parent[k]=self.parent[self.parent[k]];k=self.parent[k]
        return k
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)


def assign_groups(trajectories, namespaces, ood, seed, task_only_datasets=()):
    """A trajectory record has key,dataset,task_id,request_hash."""
    uf=UnionFind(t['key'] for t in trajectories)
    tasks={};requests={}
    for t in trajectories:
        if t['task_id'] is not None:
            identity=(namespaces.get(t['dataset'],t['dataset']),t['task_id'])
            if identity in tasks:uf.union(t['key'],tasks[identity])
            else:tasks[identity]=t['key']
        if t['dataset'] in task_only_datasets:
            if t['task_id'] is None:raise ValueError('task-only grouping requires canonical task_id')
            continue
        request=t['request_hash']
        if not isinstance(request,str) or not request:raise ValueError('missing request_hash')
        if request in requests:uf.union(t['key'],requests[request])
        else:requests[request]=t['key']
    ood_roots={uf.find(t['key']) for t in trajectories if t['dataset'] in ood}
    result={};members={}
    for t in trajectories:
        root=uf.find(t['key']);group=digest('task_group',root)
        v=int(digest(seed,group),16)/2**256
        partition='test_ood' if root in ood_roots else 'val' if v<.05 else 'test_id' if v<.1 else 'train'
        result[t['key']]=(group,partition)
        members.setdefault(group,set()).add(t['dataset'])
    return result,sum(len(ds)>1 for ds in members.values())
