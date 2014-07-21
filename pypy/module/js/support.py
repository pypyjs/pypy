
from __future__ import with_statement

from rpython.rtyper.lltypesystem import lltype, rffi
from rpython.rtyper.tool import rffi_platform
from rpython.translator.tool.cbuild import ExternalCompilationInfo


eci = ExternalCompilationInfo(
    includes=['emjs.h'],
)


CONSTANTS = [
    'EMJS_ERROR', 'EMJS_OK', 'EMJS_UNDEFINED', 'EMJS_NULL', 'EMJS_FALSE',
    'EMJS_TRUE', 'EMJS_TYPE_ERROR', 'EMJS_TYPE_UNDEFINED', 'EMJS_TYPE_BOOLEAN',
    'EMJS_TYPE_NUMBER', 'EMJS_TYPE_STRING', 'EMJS_TYPE_OBJECT',
    'EMJS_TYPE_FUNCTION',
]


class CConfig:
    _compilation_info_ = eci
    size_t = rffi_platform.SimpleType("size_t", rffi.ULONG)
    emjs_handle = rffi_platform.SimpleType("emjs_handle", rffi.LONG)
    emjs_type = rffi_platform.SimpleType("emjs_type", rffi.LONG)


for constant in CONSTANTS:
    setattr(CConfig, constant, rffi_platform.ConstantInteger(constant))


class cConfig:
    pass


for k, v in rffi_platform.configure(CConfig).items():
    setattr(cConfig, k, v)

for constant in CONSTANTS:
    locals()[constant] = getattr(cConfig, constant)
del constant, CONSTANTS

SIZE_T_TP = cConfig.size_t
EMJS_HANDLE_TP = cConfig.emjs_handle
EMJS_TYPE_TP = cConfig.emjs_type
CALLBACK_TP = rffi.CCallback([rffi.VOIDP, EMJS_HANDLE_TP], EMJS_HANDLE_TP)


def external(name, args, result, **kwds):
    return rffi.llexternal(name, args, result, compilation_info=eci, **kwds)


emjs_free = external('emjs_free',
                     [EMJS_HANDLE_TP],
                     lltype.Void)

emjs_dup = external('emjs_dup',
                    [EMJS_HANDLE_TP],
                    EMJS_HANDLE_TP)

emjs_globals = external('emjs_globals',
                        [],
                        EMJS_HANDLE_TP)

emjs_prop_get = external('emjs_prop_get',
                         [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                         EMJS_HANDLE_TP)

emjs_prop_set = external('emjs_prop_set',
                         [EMJS_HANDLE_TP, EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                         EMJS_HANDLE_TP)

emjs_prop_delete = external('emjs_prop_delete',
                            [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                            EMJS_HANDLE_TP)

emjs_prop_apply = external('emjs_prop_apply',
                           [EMJS_HANDLE_TP, EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                           EMJS_HANDLE_TP)

emjs_prop_get_int = external('emjs_prop_get_int',
                             [EMJS_HANDLE_TP, lltype.Signed],
                             EMJS_HANDLE_TP)

emjs_prop_set_int = external('emjs_prop_set_int',
                             [EMJS_HANDLE_TP, lltype.Signed, EMJS_HANDLE_TP],
                             EMJS_HANDLE_TP)

emjs_prop_delete_int = external('emjs_prop_delete_int',
                                [EMJS_HANDLE_TP, lltype.Signed],
                                EMJS_HANDLE_TP)

emjs_prop_apply_int = external('emjs_prop_apply_int',
                               [EMJS_HANDLE_TP, lltype.Signed, EMJS_HANDLE_TP],
                               EMJS_HANDLE_TP)

emjs_prop_get_str = external('emjs_prop_get_str',
                             [EMJS_HANDLE_TP, rffi.CCHARP],
                             EMJS_HANDLE_TP)

emjs_prop_set_str = external('emjs_prop_set_str',
                             [EMJS_HANDLE_TP, rffi.CCHARP, EMJS_HANDLE_TP],
                             EMJS_HANDLE_TP)

emjs_prop_delete_str = external('emjs_prop_delete_str',
                                [EMJS_HANDLE_TP, rffi.CCHARP],
                                EMJS_HANDLE_TP)

emjs_prop_apply_str = external('emjs_prop_apply_str',
                               [EMJS_HANDLE_TP, rffi.CCHARP, EMJS_HANDLE_TP],
                               EMJS_HANDLE_TP)

emjs_make_str = external('emjs_make_str',
                         [rffi.CCHARP],
                         EMJS_HANDLE_TP)

emjs_make_strn = external('emjs_make_strn',
                          [rffi.CCHARP, SIZE_T_TP],
                          EMJS_HANDLE_TP)

emjs_make_int32 = external('emjs_make_int32',
                           [lltype.Signed],
                           EMJS_HANDLE_TP)

emjs_make_double = external('emjs_make_double',
                           [rffi.DOUBLE],
                           EMJS_HANDLE_TP)

emjs_make_bool = external('emjs_make_bool',
                           [lltype.Signed],
                           EMJS_HANDLE_TP)

emjs_make_undefined = external('emjs_make_undefined',
                               [],
                               EMJS_HANDLE_TP)

emjs_make_null = external('emjs_make_null',
                          [],
                          EMJS_HANDLE_TP)

emjs_make_object = external('emjs_make_object',
                            [],
                            EMJS_HANDLE_TP)

emjs_make_array = external('emjs_make_array',
                           [lltype.Signed],
                           EMJS_HANDLE_TP)

emjs_make_callback = external('emjs_make_callback',
                              [CALLBACK_TP, rffi.VOIDP],
                              EMJS_HANDLE_TP)

emjs_get_error = external('emjs_get_error',
                          [],
                          EMJS_HANDLE_TP)

emjs_set_error = external('emjs_set_error',
                          [EMJS_HANDLE_TP],
                          lltype.Void)

emjs_clear_error = external('emjs_clear_error',
                            [],
                            lltype.Void)

emjs_eval = external('emjs_eval',
                     [rffi.CCHARP],
                     EMJS_HANDLE_TP)

emjs_apply = external('emjs_apply',
                      [EMJS_HANDLE_TP, EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                      EMJS_HANDLE_TP)

emjs_new = external('emjs_new',
                    [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                    EMJS_HANDLE_TP)

emjs_typeof = external('emjs_typeof',
                       [EMJS_HANDLE_TP],
                       EMJS_TYPE_TP)

emjs_iter_all = external('emjs_iter_all',
                         [EMJS_HANDLE_TP, CALLBACK_TP, rffi.VOIDP],
                         EMJS_HANDLE_TP)

emjs_iter_own = external('emjs_iter_own',
                         [EMJS_HANDLE_TP, CALLBACK_TP, rffi.VOIDP],
                         EMJS_HANDLE_TP)

emjs_check = external('emjs_check',
                      [EMJS_HANDLE_TP],
                      lltype.Signed)

emjs_op_eq = external('emjs_op_eq',
                      [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                      lltype.Signed)

emjs_op_neq = external('emjs_op_neq',
                       [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                       lltype.Signed)

emjs_op_equiv = external('emjs_op_equiv',
                         [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                         lltype.Signed)

emjs_op_nequiv = external('emjs_op_nequiv',
                          [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                          lltype.Signed)

emjs_op_gt = external('emjs_op_gt',
                      [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                      lltype.Signed)

emjs_op_lt = external('emjs_op_lt',
                      [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                      lltype.Signed)

emjs_op_gteq = external('emjs_op_gteq',
                        [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                        lltype.Signed)

emjs_op_lteq = external('emjs_op_lteq',
                        [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                        lltype.Signed)

emjs_op_add = external('emjs_op_add',
                       [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                       EMJS_HANDLE_TP)

emjs_op_sub = external('emjs_op_sub',
                       [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                       EMJS_HANDLE_TP)

emjs_op_mul = external('emjs_op_mul',
                       [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                       EMJS_HANDLE_TP)

emjs_op_div = external('emjs_op_div',
                       [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                       EMJS_HANDLE_TP)

emjs_op_mod = external('emjs_op_mod',
                       [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                       EMJS_HANDLE_TP)

emjs_op_uplus = external('emjs_op_uplus',
                         [EMJS_HANDLE_TP],
                         EMJS_HANDLE_TP)

emjs_op_uminus = external('emjs_op_uminus',
                          [EMJS_HANDLE_TP],
                          EMJS_HANDLE_TP)

emjs_op_bw_and = external('emjs_op_bw_and',
                          [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                          EMJS_HANDLE_TP)

emjs_op_bw_or = external('emjs_op_bw_or',
                         [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                         EMJS_HANDLE_TP)

emjs_op_bw_xor = external('emjs_op_bw_xor',
                          [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                          EMJS_HANDLE_TP)

emjs_op_bw_lshift = external('emjs_op_bw_lshift',
                             [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                             EMJS_HANDLE_TP)

emjs_op_bw_rshift = external('emjs_op_bw_rshift',
                             [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                             EMJS_HANDLE_TP)

emjs_op_bw_urshift = external('emjs_op_bw_urshift',
                              [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                              EMJS_HANDLE_TP)

emjs_op_bw_neg = external('emjs_op_bw_neg',
                          [EMJS_HANDLE_TP],
                          EMJS_HANDLE_TP)

emjs_op_in = external('emjs_op_in',
                      [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                      lltype.Signed)

emjs_op_instanceof = external('emjs_op_instanceof',
                              [EMJS_HANDLE_TP, EMJS_HANDLE_TP],
                              lltype.Signed)

emjs_length = external('emjs_length',
                       [EMJS_HANDLE_TP],
                       lltype.Signed)

emjs_to_int32 = external('emjs_to_int32',
                         [EMJS_HANDLE_TP],
                         EMJS_HANDLE_TP)

emjs_to_uint32 = external('emjs_to_uint32',
                         [EMJS_HANDLE_TP],
                         EMJS_HANDLE_TP)

emjs_to_double = external('emjs_to_double',
                          [EMJS_HANDLE_TP],
                          EMJS_HANDLE_TP)

emjs_to_string = external('emjs_to_string',
                          [EMJS_HANDLE_TP],
                          EMJS_HANDLE_TP)

emjs_read_int32 = external('emjs_read_int32',
                           [EMJS_HANDLE_TP],
                           lltype.Signed)

emjs_read_uint32 = external('emjs_read_uint32',
                            [EMJS_HANDLE_TP],
                            lltype.Unsigned)

emjs_read_double = external('emjs_read_double',
                            [EMJS_HANDLE_TP],
                            rffi.DOUBLE)

emjs_read_str = external('emjs_read_str',
                         [EMJS_HANDLE_TP, rffi.CCHARP],
                         lltype.Signed)

emjs_read_strn = external('emjs_read_strn',
                         [EMJS_HANDLE_TP, rffi.CCHARP, SIZE_T_TP],
                         lltype.Signed)
