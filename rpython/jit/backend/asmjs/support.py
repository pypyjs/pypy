
from rpython.rtyper.lltypesystem import lltype, llmemory, rffi

jitCompile = rffi.llexternal("jitCompile",
                             [rffi.CCHARP], rffi.INT)
jitReserve = rffi.llexternal("jitReserve",
                             [], rffi.INT)
jitRecompile = rffi.llexternal("jitRecompile",
                               [rffi.INT, rffi.CCHARP], lltype.Void)
jitReplace = rffi.llexternal("jitReplace",
                             [rffi.INT, rffi.INT], lltype.Void)
jitInvoke = rffi.llexternal("jitInvoke",
                            [rffi.INT, llmemory.GCREF, rffi.INT], rffi.INT)
jitFree = rffi.llexternal("jitFree",
                          [rffi.INT], lltype.Void)
jitTriggerGuard = rffi.llexternal("jitTriggerGuard",
                                  [rffi.INT], lltype.Void)
jitGuardWasTriggered = rffi.llexternal("jitGuardWasTriggered",
                                       [rffi.INT], rffi.INT)
