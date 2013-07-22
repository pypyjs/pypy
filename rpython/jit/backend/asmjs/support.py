
from rpython.rtyper.lltypesystem import lltype, llmemory, rffi

jitCompile = rffi.llexternal("jitCompile",
                             [rffi.CCHARP], rffi.INT)
jitRecompile = rffi.llexternal("jitRecompile",
                               [rffi.INT, rffi.CCHARP], lltype.Void)
jitInvoke = rffi.llexternal("jitInvoke",
                            [rffi.INT, llmemory.GCREF], llmemory.GCREF)
jitReplace = rffi.llexternal("jitReplace",
                             [rffi.INT, rffi.INT], lltype.Void)
jitFree = rffi.llexternal("jitFree",
                          [rffi.INT], lltype.Void)
