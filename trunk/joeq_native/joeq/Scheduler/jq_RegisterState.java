/*
 * jq_RegisterState.java
 *
 * Created on January 12, 2001, 8:59 AM
 *
 * @author  jwhaley
 * @version 
 */

package Scheduler;

import Assembler.x86.x86Constants;
import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Clazz.jq_DontAlign;

public class jq_RegisterState implements x86Constants, jq_DontAlign {

    // WARNING: the layout of this object should match the CONTEXT data structure
    // used in GetThreadContext/SetThreadContext.  see "winnt.h".
    
    // Used as a param in GetThreadContext/SetThreadContext
    int ContextFlags;
    // debug registers
    int Dr0, Dr1, Dr2, Dr3, Dr6, Dr7;
    // floating point
    int ControlWord, StatusWord, TagWord, ErrorOffset, ErrorSelector, DataOffset, DataSelector;
    long fp0_L; short fp0_H;  // fp are 80 bits, so it is split across two fields.
    long fp1_L; short fp1_H;
    long fp2_L; short fp2_H;
    long fp3_L; short fp3_H;
    long fp4_L; short fp4_H;
    long fp5_L; short fp5_H;
    long fp6_L; short fp6_H;
    long fp7_L; short fp7_H;
    int Cr0NpxState;
    // segment registers
    int SegGs, SegFs, SegEs, SegDs;
    // integer registers
    int Edi, Esi, Ebx, Edx, Ecx, Eax;
    // control registers
    int Ebp, Eip, SegCs, EFlags, Esp, SegSs;
    
    public jq_RegisterState() {
        ControlWord = 0x027f;
        StatusWord = 0x4000;
        TagWord = 0xffff;
    }
    
    public static final int CONTEXT_i386               = 0x00010000;
    public static final int CONTEXT_CONTROL            = (CONTEXT_i386 | 0x00000001); // SS:SP, CS:IP, FLAGS, BP
    public static final int CONTEXT_INTEGER            = (CONTEXT_i386 | 0x00000002); // AX, BX, CX, DX, SI, DI
    public static final int CONTEXT_SEGMENTS           = (CONTEXT_i386 | 0x00000004); // DS, ES, FS, GS
    public static final int CONTEXT_FLOATING_POINT     = (CONTEXT_i386 | 0x00000008); // 387 state
    public static final int CONTEXT_DEBUG_REGISTERS    = (CONTEXT_i386 | 0x00000010); // DB 0-3,6,7
    public static final int CONTEXT_EXTENDED_REGISTERS = (CONTEXT_i386 | 0x00000020); // cpu specific extensions
    public static final int CONTEXT_FULL = (CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_SEGMENTS);

    /*
    public static final jq_Class _class;
    public static final jq_InstanceField _eax, _ecx, _edx, _ebx, _esi, _edi, _ebp, _esp, _eip;
    public static final jq_InstanceField _eflags, _fp_state;
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LScheduler/jq_RegisterState;");
        _eax = _class.getOrCreateInstanceField("eax", "I");
        _ecx = _class.getOrCreateInstanceField("ecx", "I");
        _edx = _class.getOrCreateInstanceField("edx", "I");
        _ebx = _class.getOrCreateInstanceField("ebx", "I");
        _esi = _class.getOrCreateInstanceField("esi", "I");
        _edi = _class.getOrCreateInstanceField("edi", "I");
        _ebp = _class.getOrCreateInstanceField("ebp", "I");
        _esp = _class.getOrCreateInstanceField("esp", "I");
        _eip = _class.getOrCreateInstanceField("eip", "I");
        _eflags = _class.getOrCreateInstanceField("eflags", "I");
        _fp_state = _class.getOrCreateInstanceField("fp_state", "[B");
    }
     */
}
