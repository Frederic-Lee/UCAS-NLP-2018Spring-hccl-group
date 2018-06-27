import numpy as np
import os

#####################################
# Non-Projective-Greedy MST Algorithm
#####################################

def find_cycles(edges):
    """ Tarjan Algorithm finding Cycles in DAG
    Args:
      edges
    Returns:
      cycles

    """
    vertices = np.arange(len(edges))
    indices  = np.zeros_like(vertices) - 1
    lowlinks = np.zeros_like(vertices) - 1
    stack   = []
    instack = np.zeros_like(vertices, dtype=np.bool)
    current_index = 0
    cycles = []

    def strong_connect(vertex, current_index):
        indices[vertex]  = current_index
        lowlinks[vertex] = current_index
        stack.append(vertex)
        current_index += 1
        instack[vertex] = True

        for vertex_ in np.where(edges == vertex)[0]:
            if indices[vertex_] == -1:
                current_index = strong_connect(vertex_, current_index)
                lowlinks[vertex] = min(lowlinks[vertex], lowlinks[vertex_])
            elif instack[vertex_]:
                lowlinks[vertex] = min(lowlinks[vertex], indices[vertex_])

        if lowlinks[vertex] == indices[vertex]:
            cycle = []
            vertex_ = -1
            while vertex_ != vertex:
                vertex_ = stack.pop()
                instack[vertex_] = False
                cycle.append(vertex_)
            if len(cycle) > 1:
                cycles.append(np.array(cycle))

        return current_index

    for vertex in vertices:
        if indices[vertex] == -1:
            current_index = strong_connect(vertex, current_index)

    return cycles


def greedy(probs):
    """ Greedy Algorithm for MST
    Args:
        probs
    Returns:
        edges

    """
    edges  = np.argmax(probs, axis=1)
    cycles = True
    while cycles:
        cycles = find_cycles(edges)
        for cycle_vertices in cycles:
            # Get the best heads and their probabilities
            cycle_edges = edges[cycle_vertices]
            cycle_probs = probs[cycle_vertices, cycle_edges]
            # Get the second-best edges and their probabilities
            probs[cycle_vertices, cycle_edges] = 0
            backoff_edges = np.argmax(probs[cycle_vertices], axis=1)
            backoff_probs = probs[cycle_vertices, backoff_edges]
            probs[cycle_vertices, cycle_edges] = cycle_probs
            # Find the node in the cycle that the model is the least confident about and its probability
            new_root_in_cycle = np.argmax(backoff_probs/cycle_probs)
            new_cycle_root = cycle_vertices[new_root_in_cycle]
            # Set the new root
            probs[new_cycle_root, cycle_edges[new_root_in_cycle]] = 0
            edges[new_cycle_root] = backoff_edges[new_root_in_cycle]

    return edges


def chu_liu_edmonds(probs):
    """ Chu-Liu-Edmonds Algorithm for MST
    Args:
        probs
    Returns:
        edges

    """
    vertices = np.arange(len(probs))
    edges    = np.argmax(probs, axis=1)
    cycles   = find_cycles(edges)
    if cycles:
        # (c)
        cycle_vertices = cycles.pop()
        # (nc)
        non_cycle_vertices = np.delete(vertices, cycle_vertices)
        # (c)
        cycle_edges = edges[cycle_vertices]
        # get rid of cycle nodes
        # (nc x nc)
        non_cycle_probs = np.array(probs[non_cycle_vertices,:][:,non_cycle_vertices])
        # add a node representing the cycle
        # (nc+1 x nc+1)
        non_cycle_probs = np.pad(non_cycle_probs, [[0,1], [0,1]], 'constant')
        # probabilities of heads outside the cycle
        # (c x nc) / (c x 1) = (c x nc)
        backoff_cycle_probs = probs[cycle_vertices][:,non_cycle_vertices] / probs [cycle_vertices,cycle_edges][:,None]
        # probability of a node inside the cycle depending on something outside the cycle
        # max_0(c x nc) = (nc)
        non_cycle_probs[-1,:-1] = np.max(backoff_cycle_probs, axis=0)
        # probability of a node outside the cycle depending on something inside the cycle
        # max_1(nc x c) = (nc)
        non_cycle_probs[:-1,-1] = np.max(probs[non_cycle_vertices][:,cycle_vertices], axis=1)
        # (nc+1)
        non_cycle_edges = chu_liu_edmonds(non_cycle_probs)
        # This is the best source vertex into the cycle
        non_cycle_root, non_cycle_edges = non_cycle_edges[-1], non_cycle_edges[:-1] # in (nc)
        source_vertex = non_cycle_vertices[non_cycle_root] # in (v)
        # This is the vertex in the cycle we want to change
        cycle_root = np.argmax(backoff_cycle_probs[:,non_cycle_root]) # in (c)
        target_vertex = cycle_vertices[cycle_root] # in (v)
        edges[target_vertex] = source_vertex
        # update edges with any other changes
        mask = np.where(non_cycle_edges < len(non_cycle_probs)-1)
        edges[non_cycle_vertices[mask]] = non_cycle_vertices[non_cycle_edges[mask]]
        mask = np.where(non_cycle_edges == len(non_cycle_probs)-1)
        edges[non_cycle_vertices[mask]] = cycle_vertices[np.argmax(probs[non_cycle_vertices][:,cycle_vertices], axis=1)]

    return edges


def find_roots(edges):
    """Finding roots
    Returns:
        root
    """
    return np.where(edges[1:] == 0)[0]+1


def make_root(probs, root):
    """Making root
    Returns:
        probs
    """
    probs = np.array(probs)
    probs[1:,0] = 0
    probs[root,:] = 0
    probs[root,0] = 1
    probs /= np.sum(probs, axis=1, keepdims=True)

    return probs


def score_edges(probs, edges):
    """Scoring edges
    Returns:
        score
    """
    return np.sum(np.log(probs[np.arange(1,len(probs)), edges[1:]]))


def non_projective_greedy(probs, mst="greedy"):
    """Non-projective Algorithm
    Args:
        probs
        mst: greedy or chu-liu-edmonds
    Returns:
        edges
    """
    probs *= 1-np.eye(len(probs)).astype(np.float32)
    probs[0] = 0
    probs[0,0] = 1
    probs /= np.sum(probs, axis=1, keepdims=True)

    if mst is "greedy":
        edges = greedy(probs)
    else:
        edges = chu_liu_edmonds(probs)
    roots = find_roots(edges)
    best_edges = edges
    best_score = -np.inf
    if len(roots) > 1:
        for root in roots:
            probs_ = make_root(probs, root)
            if mst is "greedy":
                edges_ = greedy(probs_)
            else:
                edges_ = chu_liu_edmonds(probs_)
            score = score_edges(probs_, edges_)
            if score > best_score:
                best_edges = edges_
                best_score = score

    return best_edges


######################################
# Non-Projective-Edmonds MST Algorithm
######################################

def non_projective_edmonds(probs):
    """Non-projective Chu-Liu-Edmonds Algorithm
    Args:
        probs
        mst: greedy or chu-liu-edmonds
        Returns:
        edges
    """
    def find_cycle(par):
        added = np.zeros([length], np.bool)
        added[0] = True
        cycle = set()
        findcycle = False
        for i in range(1, length):
            if findcycle:
                break
            if added[i] or not curr_nodes[i]:
                continue

            tmp_cycle = set()
            tmp_cycle.add(i)
            added[i]  = True
            findcycle = True
            l = i

            while par[l] not in tmp_cycle:
                l = par[l]
                if added[l]:
                    findcycle = False
                    break
                added[l] = True
                tmp_cycle.add(l)

            if findcycle:
                lorg = l
                cycle.add(lorg)
                l = par[lorg]
                while l != lorg:
                    cycle.add(l)
                    l = par[l]
                break

        return findcycle, cycle

    def chuLiuEdmonds():
        # find best edge-set
        par = np.zeros([length], dtype=np.int32)
        par[0] = -1
        for i in range(1, length):
            if curr_nodes[i]:
                max_score = score_matrix[0, i]
                par[i] = 0
                for j in range(1, length):
                    if j == i or not curr_nodes[j]:
                        continue
                    new_score = score_matrix[j, i]
                    if new_score > max_score:
                        max_score = new_score
                        par[i] = j

        # find a cycle
        findcycle, cycle = find_cycle(par)

        # if no cycle, get all edges and return them.
        if not findcycle:
            final_edges[0] = -1
            for i in range(1, length):
                if not curr_nodes[i]:
                    continue
                pr = oldI[par[i], i]
                ch = oldO[par[i], i]
                final_edges[ch] = pr

            return

        cyc_len = len(cycle)
        cyc_weight = 0.0
        cyc_nodes = np.zeros([cyc_len], dtype=np.int32)
        id = 0
        for cyc_node in cycle:
            cyc_nodes[id] = cyc_node
            id += 1
            cyc_weight += score_matrix[par[cyc_node], cyc_node]

        rep = cyc_nodes[0]
        for i in range(length):
            if not curr_nodes[i] or i in cycle:
                continue

            max1 = float("-inf")
            wh1 = -1
            max2 = float("-inf")
            wh2 = -1
            for j in range(cyc_len):
                j1 = cyc_nodes[j]
                if score_matrix[j1, i] > max1:
                    max1 = score_matrix[j1, i]
                    wh1 = j1
                scr = cyc_weight + score_matrix[i, j1] - score_matrix[par[j1], j1]
                if scr > max2:
                    max2 = scr
                    wh2 = j1

            score_matrix[rep, i] = max1
            oldI[rep, i] = oldI[wh1, i]
            oldO[rep, i] = oldO[wh1, i]
            score_matrix[i, rep] = max2
            oldO[i, rep] = oldO[i, wh2]
            oldI[i, rep] = oldI[i, wh2]

        rep_cons = []
        for i in range(cyc_len):
            rep_cons.append(set())
            cyc_node = cyc_nodes[i]
            for cc in reps[cyc_node]:
                rep_cons[i].add(cc)
        for i in range(1, cyc_len):
            cyc_node = cyc_nodes[i]
            curr_nodes[cyc_node] = False
            for cc in reps[cyc_node]:
                reps[rep].add(cc)

        chuLiuEdmonds()

        # check each node in cycle, if one of its representatives is a key in the final_edges, it is the one.
        found = False
        wh = -1
        for i in range(cyc_len):
            for repc in rep_cons[i]:
                if repc in final_edges:
                    wh = cyc_nodes[i]
                    found = True
                    break
            if found:
                break

        l = par[wh]
        while l != wh:
            ch = oldO[par[l], l]
            pr = oldI[par[l], l]
            final_edges[ch] = pr
            l = par[l]

    # reset probs
    probs *= 1-np.eye(len(probs)).astype(np.float32)
    probs[0] = 0
    probs[0,0] = 1
    probs /= np.sum(probs, axis=1, keepdims=True)

    # get original score matrix
    length = len(probs)
    orig_score_matrix = probs
    score_matrix = np.array(orig_score_matrix, copy=True)

    oldI = np.zeros([length, length], dtype=np.int32)
    oldO = np.zeros([length, length], dtype=np.int32)
    curr_nodes = np.zeros([length], dtype=np.bool)
    reps = []
    for s in range(length):
        orig_score_matrix[s, s] = 0.0
        score_matrix[s, s] = 0.0
        curr_nodes[s] = True
        reps.append(set())
        reps[s].add(s)
        for t in range(s + 1, length):
            oldI[s, t] = s
            oldO[s, t] = t
            oldI[t, s] = t
            oldO[t, s] = s

    # chu-liu-edmonds
    final_edges = dict()
    chuLiuEdmonds()

    par = np.zeros([length], np.int32)
    for ch, pr in final_edges.items():
        par[ch] = pr
    par[0] = 0

    return par

##########################
# Projective MST Algorithm
##########################

def projective_eisner(scores, gold=None):
    """ Projective Eisner's algorithm """
    nr, nc = np.shape(scores)
    if nr != nc:
        raise ValueError("scores must be a squared matrix with nw+1 rows")

    N = nr - 1 # Number of words (excluding root).

    # Initialize CKY table.
    complete = np.zeros([N+1, N+1, 2]) # s, t, direction (right=1).
    incomplete = np.zeros([N+1, N+1, 2]) # s, t, direction (right=1).
    complete_backtrack = -np.ones([N+1, N+1, 2], dtype=int) # s, t, direction (right=1).
    incomplete_backtrack = -np.ones([N+1, N+1, 2], dtype=int) # s, t, direction (right=1).

    incomplete[0, :, 0] -= np.inf

    # Loop from smaller items to larger items.
    for k in range(1,N+1):
        for s in range(N-k+1):
            t = s+k

            # First, create incomplete items.
            # left tree
            incomplete_vals0 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[t, s] + (0.0 if gold is not None and gold[s]==t else 1.0)
            incomplete[s, t, 0] = np.max(incomplete_vals0)
            incomplete_backtrack[s, t, 0] = s + np.argmax(incomplete_vals0)
            # right tree
            incomplete_vals1 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[s, t] + (0.0 if gold is not None and gold[t]==s else 1.0)
            incomplete[s, t, 1] = np.max(incomplete_vals1)
            incomplete_backtrack[s, t, 1] = s + np.argmax(incomplete_vals1)

            # Second, create complete items.
            # left tree
            complete_vals0 = complete[s, s:t, 0] + incomplete[s:t, t, 0]
            complete[s, t, 0] = np.max(complete_vals0)
            complete_backtrack[s, t, 0] = s + np.argmax(complete_vals0)
            # right tree
            complete_vals1 = incomplete[s, (s+1):(t+1), 1] + complete[(s+1):(t+1), t, 1]
            complete[s, t, 1] = np.max(complete_vals1)
            complete_backtrack[s, t, 1] = s + 1 + np.argmax(complete_vals1)

    value = complete[0][N][1]
    heads = -np.ones(N+1, dtype=int)
    backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, N, 1, 1, heads)

    value_proj = 0.0
    for m in range(1,N+1):
        h = heads[m]
        value_proj += scores[h,m]

    return heads


def backtrack_eisner(incomplete_backtrack, complete_backtrack, s, t, direction, complete, heads):
    """
    Backtracking step in Eisner's algorithm.
    - incomplete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *incomplete* spans.
    - complete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *complete* spans.
    - s is the current start of the span
    - t is the current end of the span
    - direction is 0 (left attachment) or 1 (right attachment)
    - complete is 1 if the current span is complete, and 0 otherwise
    - heads is a (NW+1)-sized numpy array of integers which is a placeholder for storing the
    head of each word.

    """
    if s == t:
        return
    if complete:
        r = complete_backtrack[s][t][direction]
        if direction == 0:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads)
            return
        else:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads)
            return
    else:
        r = incomplete_backtrack[s][t][direction]
        if direction == 0:
            heads[s] = t
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
            return
        else:
            heads[t] = s
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
            return


#***************************************************************
if __name__ == '__main__':
    def softmax(x):
        x -= np.max(x, axis=1, keepdims=True)
        x = np.exp(x)
        return x / np.sum(x, axis=1, keepdims=True)
    probs_ = softmax(np.random.randn(20,20))
    probs = probs_ * (1-np.eye(len(probs_)).astype(np.float32))
    probs[0] = 0
    probs[0,0] = 1
    probs /= np.sum(probs, axis=1, keepdims=True)

    edges = greedy(probs)
    roots = find_roots(edges)
    print(edges)
    print(roots)

    edges = chu_liu_edmonds(probs)
    roots = find_roots(edges)
    print(edges)
    print(roots)
    """
    best_edges = edges
    best_score = -np.inf
    if len(roots) > 1:
        for root in roots:
            probs_ = make_root(probs, root)
            edges_ = nonprojective(probs_)
            score = score_edges(probs_, edges_)
            if score > best_score:
                best_edges = edges_
                best_score = score
    edges = best_edges
    print(edges)
    print(find_roots(edges))
    """

    edges = non_projective_edmonds(probs)
    roots = find_roots(edges)
    print(edges)
    print(roots)
    """
    best_edges = edges
    best_score = -np.inf
    if len(roots) > 1:
        for root in roots:
            probs_ = make_root(probs, root)
            edges_ = non_projective_edmonds(probs_)
            score = score_edges(probs_, edges_)
            if score > best_score:
                best_edges = edges_
                best_score = score
    edges = best_edges
    print(edges)
    print(find_roots(edges))
    """

    edges = projective_eisner(probs_)
    roots = find_roots(edges)
    print(edges)
    print(roots)
