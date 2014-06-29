import heapq

from rpython.rtyper import rmodel
from rpython.rtyper.lltypesystem import lltype, llmemory
from rpython.memory.gctransform.framework import sizeofaddr
from rpython.rtyper.rbuiltin import gen_cast
from rpython.flowspace.model import copygraph

from rpython.memory.gctransform.transform import GCTransformer
from rpython.memory.gctransform.shadowstack import (
    ShadowStackFrameworkGCTransformer,
)

# Test cases to add:
#   * diamond flow where var comes in twice into difference variables


class OptzShadowStackFrameworkGCTransformer(ShadowStackFrameworkGCTransformer):
    """Optimizing ShadowStack GC Transformer.

    This GCTransformer implements the same approach as the base ShadowStack
    trasform, maintaining a separate "root stack" of live gc objects that is
    updated at each potential GC point.  The difference is that it works a
    lot harder at trying to avoid redundant updates of the root stack.

    The idea is to find all potential GC points in a function and analyse the
    flow of control between them.  If we can prove that all paths to a given
    GC point will already have written a particular value to the stack, there
    is no need to write it again.
    """

    def __init__(self, translator):
        super(OptzShadowStackFrameworkGCTransformer, self).__init__(translator)
        self.analyzer = OptzAnalysisGCTransformer(self)

    def transform_graph(self, graph):
        # Do an initial analysis pass to decide on a stack management strategy.
        self.strategy = self.analyzer.analyze_graph(graph)
        # Skip the empty push event inserted at seqnum 0.
        # XXX TODO: we could do an initial increment here.
        assert self.strategy.pushes[0].incr == 0
        assert self.strategy.pushes[0].writes == []
        assert self.strategy.pushes[0].decr == 0
        # Traverse the graph and insert the necessary operations.
        self.push_seqnum = 1
        supercls = super(OptzShadowStackFrameworkGCTransformer, self)
        supercls.transform_graph(graph)
        # Skip the empty push events inserted at seqnum -1 and -2.
        # XXX TODO: we could do a final decrement here.
        assert self.push_seqnum == len(self.strategy.pushes) - 2
        assert self.strategy.pushes[-1].incr == 0
        assert self.strategy.pushes[-1].writes == []
        assert self.strategy.pushes[-1].decr == 0
        assert self.strategy.pushes[-2].incr == 0
        assert self.strategy.pushes[-2].writes == []
        assert self.strategy.pushes[-2].decr == 0

    def push_roots(self, hop, keep_current_args=False):
        push = self.strategy.pushes[self.push_seqnum]
        # Sanity-check that we've got the right push for these vars.
        livevars = self.get_livevars_for_roots(hop, keep_current_args)
        assert push.gcpoint.livevars == livevars
        # Increment the top-of-stack, which also loads the address.
        c_incr = rmodel.inputconst(lltype.Signed, push.incr)
        rst_addr = hop.genop("direct_call", [self.incr_stack_ptr, c_incr],
                             resulttype=llmemory.Address)
        # Write out the required stack slots.
        # Some may be None, indicating that we should write a NULL pointer.
        for slot in push.writes:
            var = push.stack[slot]
            if var is not None:
                v_adr = gen_cast(hop.llops, llmemory.Address, var)
            else:
                null = rmodel.inputconst(lltype.Signed, 0)
                v_adr = gen_cast(hop.llops, llmemory.Address, null)
            # Loading rst_addr gives the old top-of-stack, which may not
            # correspond to the bottom of our frame.  Adjust as necessary.
            relslot = slot + (push.incr - len(push.stack))
            c_slot = rmodel.inputconst(lltype.Signed, relslot * sizeofaddr)
            hop.genop("raw_store", [rst_addr, c_slot, v_adr])
        # The following inserts sanity-checking assertions to ensure that the
        # stack is in the desired state; uncomment to assist debugging.
        #for slot, var in enumerate(push.stack):
        #    if var is not None:
        #        v_adr = gen_cast(hop.llops, llmemory.Address, var)
        #    else:
        #        null = rmodel.inputconst(lltype.Signed, 0)
        #        v_adr = gen_cast(hop.llops, llmemory.Address, null)
        #    relslot = slot + (push.incr - len(push.stack))
        #    c_slot = rmodel.inputconst(lltype.Signed, relslot * sizeofaddr)
        #    v_fromstack = hop.genop("raw_load", [rst_addr, c_slot],
        #                            resulttype=llmemory.Address)
        #    v_eq = hop.genop("ptr_eq", [v_adr, v_fromstack],
        #                     resulttype=lltype.Bool)
        #    c_errmsg = rmodel.inputconst(lltype.Void,
        #                                 "gc stack incorrect for %s" % (var,))
        #    hop.genop("debug_assert", [v_eq, c_errmsg])
        # That's it!
        self.num_pushs += len(push.writes)
        return push.livevars

    def pop_roots(self, hop, livevars):
        push = self.strategy.pushes[self.push_seqnum]
        # Sanity-check that we've got the right push for these vars.
        assert push.livevars == livevars
        # Decrement the top-of-stack, which also loads the address.
        c_decr = rmodel.inputconst(lltype.Signed, push.decr)
        rst_addr = hop.genop("direct_call", [self.decr_stack_ptr, c_decr],
                             resulttype=llmemory.Address)
        # For moving collectors, reload all roots into the local variables.
        # Some of these may be redundant, e.g. the loaded value is never used
        # before being clobbered by a subsequent reload into that variable.
        # We don't bother trying to eliminate these, as any halfway decent
        # compiler will eliminate them itself via liveness analysis.
        if self.gcdata.gc.moving_gc:
            for var in livevars:
                slot = push.varlocs[var]
                # Loading rst_addr gives the new top-of-stack, which may not
                # correspond to the bottom of our frame.  Adjust as necessary.
                relslot = slot + (push.decr - len(push.stack))
                c_slot = rmodel.inputconst(lltype.Signed, relslot * sizeofaddr)
                v_newaddr = hop.genop("raw_load", [rst_addr, c_slot],
                                      resulttype=llmemory.Address)
                hop.genop("gc_reload_possibly_moved", [v_newaddr, var])
        # That's it!  Move on to the next push in the graph.
        self.push_seqnum += 1


class OptzAnalysisGCTransformer(ShadowStackFrameworkGCTransformer):
    """Dummy GCTransformer that analyzes control flow between GC points.

    This class is a bit of a hack to re-use the infrastructure of the
    GCTransformer for an analysis phase.  Rather than re-writing the given
    graph to insert GC management operations, it simply notes each GC point
    as it is traversed so that the flow graph between then can be constructed.

    Doing a preliminary analysis pass allows for whole-graph optimization of
    the root stack layout, which can lead to more opportunities to re-use
    stack writes between GC points.

    Instaces of this class provide an "analyze_graph" method which will take
    a function graph, and return a RootStackStrategy describing how to manage
    the root stack.  A second pass can then be done to transform the graph in
    accordance with this strategy.
    """

    def __init__(self, parenttransform):
        # Due to global state, we can't create a whole second instance of the
        # FrameworkGCTransformer class.  Instead we proxy attributes from the
        # parent as needed.
        GCTransformer.__init__(self, parenttransform.translator)
        self.parenttransform = parenttransform
        self.num_pushs = 0

    def __getattr__(self, nm):
        return getattr(self.parenttransform, nm)

    def analyze_graph(self, orig_graph):
        print "ANALYZE"
        print orig_graph
        # Operate on a copy so as not to accidentally change the input graph.
        graph = copygraph(orig_graph, shallow=True)
        if orig_graph in self.parenttransform.seen_graphs:
            self.seen_graphs.add(graph)
        if orig_graph in self.parenttransform.minimal_transform:
            self.minimal_transform.add(graph)
        # Do a fake transform pass to collect details on the GC events.
        self.gcpointgraph = GCPointGraph(graph)
        self.transform_graph(graph)
        self.gcpointgraph.finalize()
        # Find the lowest-cost stack management strategy.
        strategies = list(self._generate_strategies(self.gcpointgraph))
        print "--strategies--"
        for s in strategies:
            print "%s: %d" % (s.__class__.__name__, s.estimated_cost)
        strategy = min(strategies, key=lambda s: s.estimated_cost)
        print "DONE:", strategy.__class__.__name__
        return strategy

    def transform_block(self, block, is_borrowed):
        self.curr_block = block
        supercls = super(OptzAnalysisGCTransformer, self)
        supercls.transform_block(block, is_borrowed)

    def push_roots(self, hop, keep_current_args=False):
        # Each call to this method represents a distinct point at which
        # garbage collection may occur.  Note it for later analysis.
        livevars = self.get_livevars_for_roots(hop, keep_current_args)
        self.gcpointgraph.add_gcpoint(self.curr_block, livevars)
        return livevars

    def pop_roots(self, hop, livevars):
        pass

    def _generate_strategies(self, gcpointgraph):
        """Generate candidate RootStackStragegy objects for a GCPointGraph.

        This method applies several different concrete RootStackStrategy
        subclasses, yielding each in turn.  The minimal such strategy is
        the best one to use for the given graph.
        """
        yield DenseFirstFitStrategy(gcpointgraph)
        yield SparseFirstFitStrategy(gcpointgraph)
        yield DenseGreedyStrategy(gcpointgraph)
        yield SparseGreedyStrategy(gcpointgraph)
        yield PeepholeStrategy(gcpointgraph)


class GCPoint(object):
    """Object representing a control-flow point at which GC may occur.

    Initial analysis of a flow graph will produce a corresponding graph of
    GCPoint objects, representing the flow between different points at which
    garbage collection may occur.  Each is annotated with a sequence number,
    its owning block, and the set of variables live at that point.
    """

    def __init__(self, seqnum, block, livevars):
        self.seqnum = seqnum
        self.block = block
        self.livevars = list(livevars)


class GCPointGraph(object):
    """Object representing control flow between potential gc operations.

    This class holds a list of GCPoint objects, each representing a point
    in the control flow at which garbage collection might occur.  The points
    have a linear ordering corresponding to their traversal order in the
    flow graph, and successor/predecessor relationships representing control
    flow between them.

    The 0th GCPoint represents the entrypoint of the flow graph, while the
    -2nd and -1st GCPoints represents its two exit blocks.  These all have an
    empty set of live variables but exist to make reasoning simpler.
    """

    def __init__(self, graph):
        self.graph = graph
        self.gcpoints = []
        self.successors = {}
        self.predecessors = {}
        self.block_entries = {}
        # Create the entry gcpoint.
        self.add_gcpoint(graph.startblock, [])

    def add_gcpoint(self, block, livevars):
        """Mark a garbage-collection point in the given block.

        We assume that blocks are traversed in order, so that the first time
        each block is seen indicates the first gcpoint in that block.
        """
        gcpoint = GCPoint(len(self.gcpoints), block, livevars)
        self.gcpoints.append(gcpoint)
        self.successors[gcpoint.seqnum] = []
        self.predecessors[gcpoint.seqnum] = []
        if block not in self.block_entries:
            self.block_entries[block] = gcpoint

    def finalize(self):
        """Finalize the graph structure from the marked gcpoints."""
        # Create the exit gcpoints for return and except blocks.
        assert self.graph.returnblock not in self.block_entries
        assert self.graph.exceptblock not in self.block_entries
        self.add_gcpoint(self.graph.returnblock, [])
        self.add_gcpoint(self.graph.exceptblock, [])
        # Walk the gcpoints to annotate each with its successors.
        # (Predecessors are handled reflexively when adding successors).
        for block, gcpoint in self.block_entries.iteritems():
            try:
                next_gcpoint = self.gcpoints[gcpoint.seqnum + 1]
            except IndexError:
                continue
            # Step forward over each successor in the same block.
            while next_gcpoint.block is block:
                self._add_successor(gcpoint, next_gcpoint)
                gcpoint = next_gcpoint
                next_gcpoint = self.gcpoints[gcpoint.seqnum + 1]
            # The final gcpoint in the block must link to the entry
            # gcpoint of the successor blocks.
            for next_block, varmap in self._find_successor_blocks(block):
                next_gcpoint = self.block_entries[next_block]
                self._add_successor(gcpoint, next_gcpoint, varmap)

    def get_successors(self, gcpoint):
        for succ, varmap in self.successors[gcpoint.seqnum]:
            yield succ, varmap

    def get_predecessors(self, gcpoint):
        for pred, varmap in self.predecessors[gcpoint.seqnum]:
            yield pred, varmap

    def _add_successor(self, prev_gcpoint, next_gcpoint, varmap=None):
        """Mark a predecessor/successor relationship between two gcpoints.

        If given, 'varmap' is a dict mapping variables in the successor
        block to their source variables in the predecessor block.
        """
        if varmap is None:
            # Two points in the same block; varmap should be only
            # the live vars that are shared between them.
            varmap = {}
            for var in next_gcpoint.livevars:
                if var in prev_gcpoint.livevars:
                    varmap[var] = var
        self.successors[prev_gcpoint.seqnum].append((next_gcpoint, varmap))
        self.predecessors[next_gcpoint.seqnum].append((prev_gcpoint, varmap))

    def _find_successor_blocks(self, block, varmap=None, seen=None):
        """Find blocks following the given block that contain a gcpoint.

        This method walks forward from the given block, to find successor
        blocks that contain gcpoints.  It effectively "looks through" any
        successor blocks where no GC takes place.

        It yields pairs (block, varmap) where the varmap is a dict mapping
        variables in the successor block to their source variables in the
        input block.  This is necessary for tracking live objects as they are
        renamed across blocks.
        """
        # Avoid visiting (block, varmap) pairs we've already seen.
        if seen is None:
            seen = {}
        if block not in seen:
            seen[block] = []
        if varmap in seen[block]:
            return
        seen[block].append(varmap)
        # Follow each outgoing link from the block.
        for link in block.exits:
            target = link.target
            # Record the mapping of variables done by this link.
            next_varmap = {}
            for i, next_var in enumerate(target.inputargs):
                if varmap is None:
                    next_varmap[next_var] = link.args[i]
                else:
                    try:
                        next_varmap[next_var] = varmap[link.args[i]]
                    except KeyError:
                        pass
            # If the target contains a gcpoint, yield it directly.
            if target in self.block_entries:
                yield target, next_varmap
                continue
            # Otherwise recurse through it, into its successors.
            for res in self._find_successor_blocks(target, next_varmap, seen):
                yield res


class RootStackPush(object):
    """Object representing the adjustments to make for a RootStackPush.

    This object corresponds to a RootStackPush, and gives the particular set
    of root-stack manipulations to be performance to execute that push.  Its
    fields are:

        stack:     list mapping stack slot nums to their contents at that point
        varlocs:   dict mapping variables to their location on the stack
        incr:      the amount by which we must increment the stack pointer
        writes:    the stack slots that we must write out to memory
        decr:      the amount by which we must decerement the stack pointer

    Some stack positions may contain None rather than a variable, indicating
    that a NULL pointer should be written to the stack at that location.
    """

    def __init__(self, gcpoint):
        self.gcpoint = gcpoint
        self.stack = []
        self.varlocs = {}
        self.incr = 0
        self.writes = []
        self.decr = 0

    @property
    def livevars(self):
        return self.gcpoint.livevars

    @property
    def seqnum(self):
        return self.gcpoint.seqnum


class RootStackStrategy(object):
    """Object representing a graph of root-stack push operations.

    This class maps a graph of GCPoint objects to a corresponding graph
    of RootStackPush objects.  Each RootStackPush indicates the concrete
    actions to be taken at that point in order to record all live variables
    in the root-stack.

    Subclasses of RootStackStrategy will implement different strategies
    for managing the stack, with potentially differing costs.  Each has
    an attribute 'estimated_cost' to give the approximate overhead imposed
    by the strategy for the target graph.
    """

    def __init__(self, gcpointgraph):
        self.gcpointgraph = gcpointgraph
        self.pushes = []
        for gcpoint in gcpointgraph.gcpoints:
            self.pushes.append(RootStackPush(gcpoint))
        self.estimated_cost = 0
        self.allocate_stack_positions()
        self.calculate_stack_operations()
        self.calculate_estimated_cost()
        self.sanity_check()

    def get_successors(self, push):
        """Yield (push, varmap) pairs for successor gc points."""
        for succ, varmap in self.gcpointgraph.get_successors(push.gcpoint):
            yield self.pushes[succ.seqnum], varmap

    def get_predecessors(self, push):
        """Yield (push, varmap) pairs for predecessor gc points."""
        for pred, varmap in self.gcpointgraph.get_predecessors(push.gcpoint):
            yield self.pushes[pred.seqnum], varmap

    def allocate_stack_positions(self):
        """Select stack layout for each push, populating 'stack' and 'varlocs'.

        Subclasses should override this method to implement a particular
        strategy for assigning vars to stack locations.
        """
        raise NotImplementedError

    def calculate_stack_operations(self):
        """Calculate stack operations for each push, based on stack layout.

        This method populates the 'incr', 'decr', and 'writes' fields based
        on the allocated stack layout at each point.  It tries to avoid doing
        redundant work e.g. writing variables to the stack if they're already
        guaranteed to be there, or adjusting the root-stack-top if it's already
        at the right size.
        """
        for push in self.pushes:
            predecessors = list(self.get_predecessors(push))
            # Sanity-check: all points must have a predecessor except
            # for the entry block, and maybe the exceptblock.
            if 0 < push.seqnum < len(self.pushes) - 1:
                assert len(predecessors) > 0
            # Find the necessary stack increment/decrement.
            # To begin we assume that each resets the stack size to zero.
            # We can adjust this later after all else is assigned.
            if len(push.stack) > 0:
                push.incr = push.decr = len(push.stack)
            # Find slots that have to be written to the stack.
            # We can avoid writing a var if all predecessors already wrote it.
            if not predecessors:
                # No predecessors, we have to write everything.
                for slot, var in enumerate(push.stack):
                    push.writes.append(slot)
            else:
                # See which slots we can re-use from predecessors.
                for slot, var in enumerate(push.stack):
                    for pred_push, varmap in predecessors:
                        # It might be a new var, or be in a different slot.
                        try:
                            if var is None:
                                pred_var = None
                            else:
                                pred_var = varmap[var]
                            if pred_push.stack[slot] is not pred_var:
                                raise KeyError
                        except (IndexError, KeyError):
                            # Missing in a predecessor, we have to write.
                            push.writes.append(slot)
                            break
        # Try to re-use stack increments between pushes, rather than always
        # returning it to zero.  To do this we calculate sets of predecessor
        # and successor "siblings".  If all successor siblings require the
        # same stack depth, then we can have all predecessor siblings set it
        # to that depth rather than resetting it to zero.
        done = set()
        for push in self.pushes:
            if push in done:
                continue
            # Calculate transitive sibling predecessor/successor sets,
            # using the current push as seed predecessor.
            pred_todo = set((push,))
            pred_sibs = set()
            succ_sibs = set()
            while pred_todo:
                pred = pred_todo.pop()
                pred_sibs.add(pred)
                for succ, _ in self.get_successors(pred):
                    if succ not in succ_sibs:
                        succ_sibs.add(succ)
                        for pred, _ in self.get_predecessors(succ):
                            if pred not in pred_sibs:
                                pred_todo.add(pred)
            # See if we can re-use stack increment for this sibling group.
            # XXX TODO: we can't do incr/decr at entry/exit yet, so
            # we must skip this optimization if those pushes are involved.
            succ_stack_sizes = set(len(succ.stack) for succ in succ_sibs)
            for pred in pred_sibs:
                if pred.seqnum == 0 and succ_stack_sizes:
                    succ_stack_sizes.add(-1)
            for succ in succ_sibs:
                if succ.seqnum >= len(self.pushes) - 2:
                    succ_stack_sizes.add(-1)
            if len(succ_stack_sizes) == 1:
                succ_stack_size = succ_stack_sizes.pop()
                for pred in pred_sibs:
                    pred.decr = len(pred.stack) - succ_stack_size
                for succ in succ_sibs:
                    succ.incr = 0
            done.update(pred_sibs)

    def calculate_estimated_cost(self):
        """Calculate estimated cost of this strategy.

        The cost is estimated as follows:
            * +1 for every read of root-stack-top
            * +1 for every write to root-stack-top
            * +1 for every write of a variable to the stack

        """
        self.estimated_cost = 0
        for push in self.pushes:
            # Read, and possibly incremenet, root-stack-top.
            self.estimated_cost += 1
            if push.incr != 0:
                self.estimated_cost += 1
            # Write out some number of slots to the stack.
            self.estimated_cost += len(push.writes)
            # Read, and possibly decrement, root-stack-top.
            self.estimated_cost += 1
            if push.decr != 0:
                self.estimated_cost += 1

    def sanity_check(self):
        for push in self.pushes:
            # All livevars have a location.
            assert sorted(push.varlocs.keys()) == sorted(push.livevars)
            # Each slot contains the allocated variable.
            for var, slot in push.varlocs.iteritems():
                assert push.stack[slot] == var
            for slot, var in enumerate(push.stack):
                if var is not None:
                    assert push.varlocs[var] == slot


class PeepholeStrategy(RootStackStrategy):
    """RootStackStrategy that considers each gc point independently.

    This is the default, safe and naive root-stack managment strategy, and
    is equivalent to the base shadowstack algorithm - it just plops the
    live variables onto the stack in whatever order they appear.

    It is guaranteed to be *no worse* than the default shadowstack strategy,
    and may be better due to re-using stack slots that happen to coincide
    between gc points.
    """

    def allocate_stack_positions(self):
        for push in self.pushes:
            for slot, var in enumerate(push.livevars):
                push.varlocs[var] = slot
                push.stack.append(var)


class GoodSlotAllocation(object):
    """Object representing a "good" slot allocation for a variable.

    A "good" slot allocation is one that allows at least one stack write to
    be elided.  This class tracks the set of slots that would allow such a
    thing, the (push, var) pairs at which that slot must be allocated, and
    the number of writes that can be elided by doing so.

    The final state of a GoodSlotAllocation is built up by the search
    process in BaseOptsStrategy.find_good_slot_allocation()
    """

    def __init__(self, push, var, slots):
        self.push = push
        self.var = var
        self.slots = slots
        self.score = 0
        self.linked = {push: var}

    def __lt__(self, other):
        # Try to sort GoodSlotAllocation objects in a sensible order from
        # "most promising" to "least promising".
        if self.score < other.score:
            return True
        if self.score > other.score:
            return False
        if len(self.slots) < len(other.slots):
            return True
        if len(self.slots) > len(other.slots):
            return False
        if len(self.linked) > len(other.linked):
            return True
        if len(self.linked) < len(other.linked):
            return False
        return self.push.seqnum < other.push.seqnum


class BaseOptzStrategy(RootStackStrategy):
    """Abstract base class for strategies that try optimizing allocation.

    This class provides some generic functionality for finding a "good"
    slot allocation for each variable, which can then be deployed in a variety
    of different concrete strategies.
    """

    def find_good_slot_allocation(self, push, var):
        """Find good slots to which to allocate the given variable.

        This method tries to find slots to allocate to the given variable that
        will enable good re-use of preceeding stack positions.  It returns a
        GoodSlotAllocation object giving the set of good slots, a dict mapping
        pushes to the variable that should be given the slot at that push, and
        a score indicating how many writes can be elided by this allocation.

        If there are no good slots then None is returned; any slot choice would
        be equally good (or bad) for that variable.
        """
        slots = self.find_available_slots(push, var)
        assert slots, "somehow ran out of free slots in a push"
        alloc = GoodSlotAllocation(push, var, slots)
        return self._find_good_slot_allocation_rec(push, var, alloc, set())

    def _find_good_slot_allocation_rec(self, push, var, alloc, seen):
        # Make sure we only visit each push once.
        if push in seen:
            if alloc.linked.get(push) == var:
                return alloc
            return None
        seen.add(push)
        # See which of the slots can actually be used at this push.
        good_slots = set(slot for slot in alloc.slots)
        good_slots.intersection_update(self.find_available_slots(push, var))
        if not good_slots:
            return None
        # See which slots can be usefully re-used in all predecessors.
        predecessors = {}
        try:
            for pred_push, varmap in self.get_predecessors(push):
                pred_var = varmap[var]
                # If it maps to two different things in the same predecessor,
                # we can't re-use the slot from that predecessor.
                if alloc.linked.get(pred_push, pred_var) != pred_var:
                    raise KeyError
                if predecessors.setdefault(pred_push, pred_var) != pred_var:
                    raise KeyError
                pred_slots = self.find_available_slots(pred_push, pred_var)
                good_slots.intersection_update(pred_slots)
                if not good_slots:
                    raise KeyError
        except KeyError:
            # There's a predecessor at which no slots can be re-used.
            return None
        # If there are no predecessors, we cannot re-use any slot.
        if not predecessors:
            return None
        # There is at least one slot that will allow re-use at this point!
        # From here on we will definitely return the allocation, so update
        # its search state data accordingly.
        assert good_slots
        alloc.score += 1
        alloc.linked[push] = var
        alloc.linked.update(predecessors)
        alloc.slots.intersection_update(good_slots)
        # Try to push the slots recursively back through predecessors.
        for pred_push, varmap in self.get_predecessors(push):
            pred_var = varmap[var]
            self._find_good_slot_allocation_rec(pred_push, pred_var,
                                                alloc, seen)
        # Try to push the slots recursively forward through successors.
        for succ_push, varmap in self.get_successors(push):
            for succ_var, pred_var in varmap.iteritems():
                if pred_var == var:
                    self._find_good_slot_allocation_rec(succ_push, succ_var,
                                                        alloc, seen)
                    break
        return alloc

    def find_available_slots(self, push, var):
        """Find the set of candidate slots for a var at the given push.

        If the var already has an allocated slot, this will return a one-item
        set with the allocated slot.  Otherwise it will return the set of all
        slots that are yet to be allocated at that push.
        """
        raise NotImplementedError


class BaseDenseStrategy(BaseOptzStrategy):

    def find_available_slots(self, push, var):
        slots = set()
        if var in push.livevars:
            try:
                slots.add(push.varlocs[var])
            except KeyError:
                for slot in xrange(len(push.livevars)):
                    try:
                        if push.stack[slot] is None:
                            slots.add(slot)
                    except IndexError:
                        slots.add(slot)
        return slots

    def allocate_slot(self, push, var, slot):
        assert var in push.livevars
        assert slot < len(push.livevars)
        if var in push.varlocs:
            assert push.varlocs[var] == slot
            return
        while slot >= len(push.stack):
            push.stack.append(None)
        if push.stack[slot] == var:
            return
        assert push.stack[slot] is None
        push.varlocs[var] = slot
        push.stack[slot] = var


class BaseSparseStrategy(BaseOptzStrategy):

    def __init__(self, gcpointgraph):
        gcpoints = gcpointgraph.gcpoints
        self._max_stack_size = max(len(p.livevars) for p in gcpoints)
        super(BaseSparseStrategy, self).__init__(gcpointgraph)

    def find_available_slots(self, push, var):
        slots = set()
        if var in push.livevars:
            try:
                slots.add(push.varlocs[var])
            except KeyError:
                for slot in xrange(self._max_stack_size):
                    try:
                        if push.stack[slot] is None:
                            slots.add(slot)
                    except IndexError:
                        slots.add(slot)
        return slots

    def allocate_slot(self, push, var, slot):
        assert var in push.livevars
        assert slot < self._max_stack_size
        if var in push.varlocs:
            assert push.varlocs[var] == slot
            return
        while len(push.stack) < self._max_stack_size:
            push.stack.append(None)
        if push.stack[slot] == var:
            return
        assert push.stack[slot] is None
        push.varlocs[var] = slot
        push.stack[slot] = var


class BaseFirstFitStrategy(BaseOptzStrategy):

    def allocate_stack_positions(self):
        # Walk the pushes allocating good slots as we find them.
        # Vars with no good slot are left until the end.
        pending = []
        for push in self.pushes:
            for var in push.livevars:
                if var in push.varlocs:
                    continue
                alloc = self.find_good_slot_allocation(push, var)
                if alloc is None:
                    pending.append((push, var))
                    continue
                slot = min(alloc.slots)
                for linked_push, linked_var in alloc.linked.iteritems():
                    self.allocate_slot(linked_push, linked_var, slot)
        # All remaining variables have no good slot, so we can
        # just arbitrarily place them in the lowest available slot.
        for (push, var) in pending:
            if var in push.varlocs:
                continue
            slot = 0
            while slot < len(push.stack) and push.stack[slot] is not None:
                slot += 1
            self.allocate_slot(push, var, slot)


class BaseGreedyStrategy(BaseOptzStrategy):

    def allocate_stack_positions(self):
        # Build a heapq whose head is the globally most-promising good slot
        # allocation.  We can then pop items off this queue for processing
        # one by one.  Vars with not good slot are left until last.
        pending = []
        promising = []
        for push in self.pushes:
            for var in push.livevars:
                alloc = self.find_good_slot_allocation(push, var)
                if alloc is None:
                    pending.append((push, var))
                    continue
                heapq.heappush(promising, alloc)
        # Now repeatedly pop the head of the promising queue and allocate it.
        while promising:
            alloc = heapq.heappop(promising)
            push = alloc.push
            var = alloc.var
            # Check whether it's still promising after previous allocations.
            alloc = self.find_good_slot_allocation(push, var)
            # If there are no longer any good slots, defer to later.
            if alloc is None:
                pending.append((push, var))
                continue
            # If it's not as promising as it was, move it back in the queue.
            if promising and promising[0] < alloc:
                heapq.heappush(promising, alloc)
                continue
            # It's still the greedily-best choice, so allocate it.
            slot = min(alloc.slots)
            for linked_push, linked_var in alloc.linked.iteritems():
                self.allocate_slot(linked_push, linked_var, slot)
        # All remaining variables have no good slots, so we can
        # just arbitrarily place them in the lowest available slot.
        for (push, var) in pending:
            if var in push.varlocs:
                continue
            slot = 0
            while slot < len(push.stack) and push.stack[slot] is not None:
                slot += 1
            self.allocate_slot(push, var, slot)


class DenseFirstFitStrategy(BaseDenseStrategy, BaseFirstFitStrategy):
    pass


class DenseGreedyStrategy(BaseDenseStrategy, BaseGreedyStrategy):
    pass


class SparseFirstFitStrategy(BaseSparseStrategy, BaseFirstFitStrategy):
    pass


class SparseGreedyStrategy(BaseSparseStrategy, BaseGreedyStrategy):
    pass
