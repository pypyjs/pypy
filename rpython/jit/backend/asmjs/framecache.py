
from rpython.rtyper.lltypesystem import lltype
from rpython.jit.backend.llsupport import jitframe


NULLJITFRAME = lltype.nullptr(jitframe.JITFRAME)


class JitFrameCache(object):
    """A simple LIFO cache of allocated JITFRAME objects.

    NOTE: This is a pretty awful hack.

    The performance of malloc_jitframe() is really bad when compiled under
    emscripten.  So much so that the cost of allocating a frame to call the
    JITed code will often outweigh the performance benefit of JITing in the
    first place!

    This very likely indicates an underlying performance issue with memory
    management under emscripten.  The right thing to do is to investigate and
    fix that.  The expedient thing to do is to hack around it by caching and
    re-using JITFRAME objects.

    Since each JITFRAME already has a field for pointing to another JITFRAME
    object, we can re-use that to make a simple linked-list of frames.  When
    When a frame is added to the list, we write the previous head of the list
    as its jf_forward field.
    """

    def __init__(self, cpu):
        self.cpu = cpu
        self.next_frame = NULLJITFRAME

    def allocate_jitframe(self, frame_info):
        frame = self.next_frame
        if frame == NULLJITFRAME:
            # XXX TODO: Frames are all the same size, for now...
            assert frame_info == self.cpu.assembler.frame_info
            frame = self.cpu.gc_ll_descr.malloc_jitframe(frame_info)
        else:
            self.next_frame = frame.jf_forward
            frame.jf_forward = NULLJITFRAME
        return frame

    def release_jitframe(self, frame):
        frame.jf_forward = self.next_frame
        self.next_frame = frame
