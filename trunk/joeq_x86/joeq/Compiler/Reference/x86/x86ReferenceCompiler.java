/*
 * x86ReferenceCompiler.java
 *
 * Created on December 22, 2000, 6:21 AM
 *
 * @author  jwhaley
 * @version 
 */

package Compil3r.Reference.x86;

import Allocator.Allocator;
import Allocator.ObjectLayout;
import Allocator.SimpleAllocator;
import Allocator.CodeAllocator;
import Assembler.x86.x86;
import Assembler.x86.x86Assembler;
import Assembler.x86.x86Constants;
import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Primitive;
import Clazz.jq_Array;
import Clazz.jq_Class;
import Clazz.jq_ClassFileConstants;
import Clazz.jq_Member;
import Clazz.jq_InstanceField;
import Clazz.jq_StaticField;
import Clazz.jq_Method;
import Clazz.jq_InstanceMethod;
import Clazz.jq_StaticMethod;
import Clazz.jq_CompiledCode;
import Clazz.jq_Type;
import Clazz.jq_NameAndDesc;
import Clazz.jq_TryCatchBC;
import Clazz.jq_TryCatch;
import Clazz.jq_BytecodeMap;
import Compil3r.Compil3rInterface;
import Run_Time.ExceptionDeliverer;
import Run_Time.MathSupport;
import Run_Time.Monitor;
import Run_Time.TypeCheck;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import UTF.Utf8;

import Compil3r.Analysis.BytecodeVisitor;

import jq;

import java.util.Collection;
import java.util.LinkedList;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;

public class x86ReferenceCompiler extends BytecodeVisitor implements Compil3rInterface, x86Constants, jq_ClassFileConstants, ObjectLayout {

    public static /*final*/ boolean ALWAYS_TRACE = false;
    public static /*final*/ boolean TRACE_STUBS = false;

    public static final Set TraceMethod_MethodNames = new HashSet();
    public static final Set TraceMethod_ClassNames = new HashSet();
    public static final Set TraceBytecode_MethodNames = new HashSet();
    public static final Set TraceBytecode_ClassNames = new HashSet();
    
    public final boolean TraceBytecodes;
    public final boolean TraceMethods;
    public final boolean TraceArguments;
    
    public x86ReferenceCompiler(jq_Method method) {
        super(method);
        TRACE = ALWAYS_TRACE;
        if (TraceBytecode_MethodNames.contains(method.getName().toString())) {
            TraceBytecodes = true;
            TraceMethods = true;
        } else if (TraceBytecode_ClassNames.contains(method.getDeclaringClass().getName().toString())) {
            TraceBytecodes = true;
            TraceMethods = true;
        } else if (TraceMethod_MethodNames.contains(method.getName().toString())) {
            TraceBytecodes = false;
            TraceMethods = true;
        } else if (TraceMethod_ClassNames.contains(method.getDeclaringClass().getName().toString())) {
            TraceBytecodes = false;
            TraceMethods = true;
        } else {
            TraceBytecodes = false;
            TraceMethods = false;
        }
        TraceArguments = false;
    }
    
    public String toString() {
        return "x86RC/"+jq.left(method.getName().toString(), 10);
    }
    
    private x86Assembler asm;   // Assembler to output to.
    private int n_paramwords;   // number of words used by incoming parameters.

    private int getLocalOffset(int local) {
        if (local < n_paramwords) {
            return (n_paramwords-local+1)<<2;
        } else {
            return (n_paramwords-local-1)<<2;
        }
    }
    
    public static final jq_CompiledCode generate_compile_stub(jq_Method method) {
        if (false) System.out.println("x86 Reference Compiler: generating compile stub for "+method);
        x86Assembler asm = new x86Assembler(13);
        if (TRACE_STUBS) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString("Stub compile: "+method)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitPUSH_i(Unsafe.addressOf(method));
        asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(jq_Method._compile, method, asm));
        asm.emit2_Mem(x86.JMP_m, jq_CompiledCode._entrypoint.getOffset(), EAX);
        // return generated code
        return CodeAllocator.allocateCodeBlock(method, asm.getCodeBuffer(), null, null, null);
    }
    
    // Generate code for the given method.
    public final jq_CompiledCode compile() {
        if (TRACE) System.out.println("x86 Reference Compiler: compiling "+method);
        
        // initialize stuff
        asm = new x86Assembler(bcs.length);
        jq_Type[] params = method.getParamTypes();
        n_paramwords = method.getParamWords();
        int n_localwords = method.getMaxLocals();
        jq.assert(n_paramwords <= n_localwords);
        
        // stack frame before prolog:
        // b0: FP->| caller's saved FP  |
        // ac:     | caller's locals    |
        //         |        ...         |
        // 94:     | caller's opstack   |
        //         |        ...         |
        // 80:     | pushed params      |
        //         |        ...         |
        // 74: SP->| ret addr in caller |
        
        // emit prolog
        asm.emitShort_Reg(x86.PUSH_r, EBP);         // push old FP
        asm.emit2_Reg_Reg(x86.MOV_r_r32, EBP, ESP); // set new FP
        if (n_paramwords != n_localwords)
            asm.emit2_Reg_Mem(x86.LEA, ESP, (n_paramwords-n_localwords)<<2, ESP);
        
        // stack frame after prolog:
        // b0:     | caller's saved FP  |
        // ac:     | caller's locals    |
        //         |        ...         |
        // 94:     | caller's opstack   |
        //         |        ...         |
        // 80:     | pushed params      |
        //         |        ...         |
        // 74:     | ret addr in caller |
        // 70: FP->| callee's FP (b0)   |
        // 6c:     | callee's locals    |
        //     SP->|        ...         |
        // 50:     | callee's opstack   |
        //         |        ...         |

        // print a debug message
        if (TraceMethods) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString("Entering: "+method)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        /*
        if (TraceArguments) {
            for (int i=0,j=0; i<params.length; ++i,++j) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString("Arg"+i+" type "+params[i]+": ")));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
                asm.emit2_Mem(x86.PUSH_m, getLocalOffset(j), EBP);
                if (params[i] == jq_Primitive.LONG) {
                    asm.emit2_Mem(x86.PUSH_m, getLocalOffset(++j), EBP);
                    asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(jq._hex16));
                } else if (params[i] == jq_Primitive.DOUBLE) {
                    asm.emit2_Mem(x86.PUSH_m, getLocalOffset(++j), EBP);
                    asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(jq._hex16));
                } else {
                    asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(jq._hex8));
                }
            }
        }
         */
        
        // add a sentinel value to the bottom of the opstack
        asm.emitPUSH_i(0x0000d00d);
        
        // generate code for each bytecode in order
        this.forwardTraversal();
        
        // generate exception table
        jq_TryCatchBC[] tcs_bc = method.getExceptionTable();
        jq_TryCatch[] tcs = new jq_TryCatch[tcs_bc.length];
        for (int i=0; i<tcs_bc.length; ++i) {
            jq_TryCatchBC tc_bc = tcs_bc[i];
            Integer start = new Integer(tc_bc.getStartPC());
            Integer end = new Integer(tc_bc.getEndPC());
            Integer handler = new Integer(tc_bc.getHandlerPC());
            jq_Class extype = tc_bc.getExceptionType();
            tcs[i] = new jq_TryCatch(asm.getBranchTarget(start), asm.getBranchTarget(end),
                                     asm.getBranchTarget(handler), extype);
        }
        
        // generate bytecode map
        Map m = asm.getBranchTargetMap();
        int numOfBC = m.size();
        int[] offsets = new int[numOfBC];
        int[] bcs = new int[numOfBC];
        ArrayList keySet = new ArrayList(m.keySet());
        java.util.Collections.sort(keySet);
        Iterator it = keySet.iterator();
        for (int i=0; i<numOfBC; ++i) {
            Integer bc = (Integer)it.next();
            bcs[i] = bc.intValue();
            offsets[i] = ((Integer)m.get(bc)).intValue();
        }
        jq_BytecodeMap bcm = new jq_BytecodeMap(offsets, bcs);
        
        // return generated code
        return CodeAllocator.allocateCodeBlock(method, asm.getCodeBuffer(), tcs, bcm, x86ReferenceExceptionDeliverer.INSTANCE);
    }
    
    public void visitBytecode() throws VerifyError {
        Integer loc = new Integer(i_start);
        asm.recordBranchTarget(loc);
        asm.resolveForwardBranches(loc);
        // do dispatch
        super.visitBytecode();
    }
    
    public void visitNOP() {
        super.visitNOP();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": NOP")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit1(x86.NOP);
    }
    public void visitACONST(Object s) {
        super.visitACONST(s);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": ACONST")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitPUSH_i(Unsafe.addressOf(s));
    }
    public void visitICONST(int c) {
        super.visitICONST(c);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": ICONST "+c)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitPUSH_i(c);
    }
    public void visitLCONST(long c) {
        super.visitLCONST(c);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": LCONST "+c)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitPUSH_i((int)(c>>32)); // hi
        asm.emitPUSH_i((int)c);       // lo
    }
    public void visitFCONST(float c) {
        super.visitFCONST(c);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": FCONST "+c)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitPUSH_i(Float.floatToRawIntBits(c));
    }
    public void visitDCONST(double c) {
        super.visitDCONST(c);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": DCONST "+c)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        long v = Double.doubleToRawLongBits(c);
        asm.emitPUSH_i((int)(v>>32)); // hi
        asm.emitPUSH_i((int)v);       // lo
    }
    public void visitILOAD(int i) {
        super.visitILOAD(i);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": ILOAD "+i)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Mem(x86.PUSH_m, getLocalOffset(i), EBP);
    }
    public void visitLLOAD(int i) {
        super.visitLLOAD(i);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": LLOAD "+i)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Mem(x86.PUSH_m, getLocalOffset(i), EBP);   // hi
        asm.emit2_Mem(x86.PUSH_m, getLocalOffset(i+1), EBP); // lo
    }
    public void visitFLOAD(int i) {
        super.visitFLOAD(i);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": FLOAD "+i)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Mem(x86.PUSH_m, getLocalOffset(i), EBP);
    }
    public void visitDLOAD(int i) {
        super.visitDLOAD(i);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": DLOAD "+i)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Mem(x86.PUSH_m, getLocalOffset(i), EBP);   // hi
        asm.emit2_Mem(x86.PUSH_m, getLocalOffset(i+1), EBP); // lo
    }
    public void visitALOAD(int i) {
        super.visitALOAD(i);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": ALOAD "+i)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Mem(x86.PUSH_m, getLocalOffset(i), EBP);
    }
    public void visitISTORE(int i) {
        super.visitISTORE(i);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": ISTORE "+i)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Mem(x86.POP_m, getLocalOffset(i), EBP);
    }
    public void visitLSTORE(int i) {
        super.visitLSTORE(i);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": LSTORE "+i)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Mem(x86.POP_m, getLocalOffset(i+1), EBP); // lo
        asm.emit2_Mem(x86.POP_m, getLocalOffset(i), EBP);   // hi
    }
    public void visitFSTORE(int i) {
        super.visitFSTORE(i);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": FSTORE "+i)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Mem(x86.POP_m, getLocalOffset(i), EBP);
    }
    public void visitDSTORE(int i) {
        super.visitDSTORE(i);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": DSTORE "+i)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Mem(x86.POP_m, getLocalOffset(i+1), EBP); // lo
        asm.emit2_Mem(x86.POP_m, getLocalOffset(i), EBP);   // hi
    }
    public void visitASTORE(int i) {
        super.visitASTORE(i);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": ASTORE "+i)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Mem(x86.POP_m, getLocalOffset(i), EBP);
    }
    private void ALOAD4helper() {
        asm.emitShort_Reg(x86.POP_r, EBX);   // array index
        asm.emitShort_Reg(x86.POP_r, EAX);   // array ref
        asm.emitARITH_Reg_Mem(x86.CMP_r_m32, EBX, ObjectLayout.ARRAY_LENGTH_OFFSET, EAX);
        asm.emitCJUMP_Short(x86.JB, (byte)2); asm.emit1_Imm8(x86.INT_i8, BOUNDS_EX_NUM);
        asm.emit2_Mem(x86.PUSH_m, EAX, EBX, SCALE_4, ObjectLayout.ARRAY_ELEMENT_OFFSET);
    }
    private void ALOAD8helper() {
        asm.emitShort_Reg(x86.POP_r, EBX);   // array index
        asm.emitShort_Reg(x86.POP_r, EAX);   // array ref
        asm.emitARITH_Reg_Mem(x86.CMP_r_m32, EBX, ARRAY_LENGTH_OFFSET, EAX);
        asm.emitCJUMP_Short(x86.JB, (byte)2); asm.emit1_Imm8(x86.INT_i8, BOUNDS_EX_NUM);
        asm.emit2_Mem(x86.PUSH_m, EAX, EBX, SCALE_8, ARRAY_ELEMENT_OFFSET+4); // hi
        asm.emit2_Mem(x86.PUSH_m, EAX, EBX, SCALE_8, ARRAY_ELEMENT_OFFSET  ); // lo
    }
    public void visitIALOAD() {
        super.visitIALOAD();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": IALOAD")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        ALOAD4helper();
    }
    public void visitLALOAD() {
        super.visitLALOAD();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": LALOAD")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        ALOAD8helper();
    }
    public void visitFALOAD() {
        super.visitFALOAD();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": FALOAD")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        ALOAD4helper();
    }
    public void visitDALOAD() {
        super.visitDALOAD();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": DALOAD")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        ALOAD8helper();
    }
    public void visitAALOAD() {
        super.visitAALOAD();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": AALOAD")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        ALOAD4helper();
    }
    public void visitBALOAD() {
        super.visitBALOAD();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": BALOAD")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitShort_Reg(x86.POP_r, EBX);   // array index
        asm.emitShort_Reg(x86.POP_r, EAX);   // array ref
        asm.emitARITH_Reg_Mem(x86.CMP_r_m32, EBX, ARRAY_LENGTH_OFFSET, EAX);
        asm.emitCJUMP_Short(x86.JB, (byte)2); asm.emit1_Imm8(x86.INT_i8, BOUNDS_EX_NUM);
        asm.emit3_Reg_Mem(x86.MOVSX_r_m8, ECX, EAX, EBX, SCALE_1, ARRAY_ELEMENT_OFFSET);
        asm.emitShort_Reg(x86.PUSH_r, ECX);
    }
    public void visitCALOAD() {
        super.visitCALOAD();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": CALOAD")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitShort_Reg(x86.POP_r, EBX);   // array index
        asm.emitShort_Reg(x86.POP_r, EAX);   // array ref
        asm.emitARITH_Reg_Mem(x86.CMP_r_m32, EBX, ARRAY_LENGTH_OFFSET, EAX);
        asm.emitCJUMP_Short(x86.JB, (byte)2); asm.emit1_Imm8(x86.INT_i8, BOUNDS_EX_NUM);
        asm.emit3_Reg_Mem(x86.MOVZX_r_m16, ECX, EAX, EBX, SCALE_2, ARRAY_ELEMENT_OFFSET);
        asm.emitShort_Reg(x86.PUSH_r, ECX);
    }
    public void visitSALOAD() {
        super.visitSALOAD();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": SALOAD")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitShort_Reg(x86.POP_r, EBX);   // array index
        asm.emitShort_Reg(x86.POP_r, EAX);   // array ref
        asm.emitARITH_Reg_Mem(x86.CMP_r_m32, EBX, ARRAY_LENGTH_OFFSET, EAX);
        asm.emitCJUMP_Short(x86.JB, (byte)2); asm.emit1_Imm8(x86.INT_i8, BOUNDS_EX_NUM);
        asm.emit3_Reg_Mem(x86.MOVSX_r_m16, ECX, EAX, EBX, SCALE_2, ARRAY_ELEMENT_OFFSET);
        asm.emitShort_Reg(x86.PUSH_r, ECX);
    }
    private void ASTORE2helper() {
        asm.emitShort_Reg(x86.POP_r, ECX);   // value
        asm.emitShort_Reg(x86.POP_r, EBX);   // array index
        asm.emitShort_Reg(x86.POP_r, EAX);   // array ref
        asm.emitARITH_Reg_Mem(x86.CMP_r_m32, EBX, ARRAY_LENGTH_OFFSET, EAX);
        asm.emitCJUMP_Short(x86.JB, (byte)2); asm.emit1_Imm8(x86.INT_i8, BOUNDS_EX_NUM);
        asm.emitprefix(x86.PREFIX_16BIT);
        asm.emit2_Reg_Mem(x86.MOV_m_r32, ECX, EAX, EBX, SCALE_2, ARRAY_ELEMENT_OFFSET);
    }
    private void ASTORE4helper() {
        asm.emitShort_Reg(x86.POP_r, ECX);   // value
        asm.emitShort_Reg(x86.POP_r, EBX);   // array index
        asm.emitShort_Reg(x86.POP_r, EAX);   // array ref
        asm.emitARITH_Reg_Mem(x86.CMP_r_m32, EBX, ARRAY_LENGTH_OFFSET, EAX);
        asm.emitCJUMP_Short(x86.JB, (byte)2); asm.emit1_Imm8(x86.INT_i8, BOUNDS_EX_NUM);
        asm.emit2_Reg_Mem(x86.MOV_m_r32, ECX, EAX, EBX, SCALE_4, ARRAY_ELEMENT_OFFSET);
    }
    private void ASTORE8helper() {
        asm.emitShort_Reg(x86.POP_r, ECX);   // lo value
        asm.emitShort_Reg(x86.POP_r, EDX);   // hi value
        asm.emitShort_Reg(x86.POP_r, EBX);   // array index
        asm.emitShort_Reg(x86.POP_r, EAX);   // array ref
        asm.emitARITH_Reg_Mem(x86.CMP_r_m32, EBX, ARRAY_LENGTH_OFFSET, EAX);
        asm.emitCJUMP_Short(x86.JB, (byte)2); asm.emit1_Imm8(x86.INT_i8, BOUNDS_EX_NUM);
        asm.emit2_Reg_Mem(x86.MOV_m_r32, ECX, EAX, EBX, SCALE_8, ARRAY_ELEMENT_OFFSET  ); // lo
        asm.emit2_Reg_Mem(x86.MOV_m_r32, EDX, EAX, EBX, SCALE_8, ARRAY_ELEMENT_OFFSET+4); // hi
    }
    public void visitIASTORE() {
        super.visitIASTORE();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": IASTORE")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        ASTORE4helper();
    }
    public void visitLASTORE() {
        super.visitLASTORE();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": LASTORE")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        ASTORE8helper();
    }
    public void visitFASTORE() {
        super.visitFASTORE();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": FASTORE")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        ASTORE4helper();
    }
    public void visitDASTORE() {
        super.visitDASTORE();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": DASTORE")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        ASTORE8helper();
    }
    public void visitAASTORE() {
        super.visitAASTORE();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": AASTORE")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        // call arraystorecheck
        asm.emit2_Mem(x86.PUSH_m, 0, ESP);  // push value
        asm.emit2_Mem(x86.PUSH_m, 12, ESP);  // push arrayref
        asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(TypeCheck._arrayStoreCheck));
        ASTORE4helper();
    }
    public void visitBASTORE() {
        super.visitBASTORE();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": BASTORE")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitShort_Reg(x86.POP_r, ECX);   // value
        asm.emitShort_Reg(x86.POP_r, EBX);   // array index
        asm.emitShort_Reg(x86.POP_r, EAX);   // array ref
        asm.emitARITH_Reg_Mem(x86.CMP_r_m32, EBX, ARRAY_LENGTH_OFFSET, EAX);
        asm.emitCJUMP_Short(x86.JB, (byte)2); asm.emit1_Imm8(x86.INT_i8, BOUNDS_EX_NUM);
        asm.emit2_Reg_Mem(x86.MOV_m_r8, ECX, EAX, EBX, SCALE_1, ARRAY_ELEMENT_OFFSET);
    }
    public void visitCASTORE() {
        super.visitCASTORE();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": CASTORE")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        ASTORE2helper();
    }
    public void visitSASTORE() {
        super.visitSASTORE();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": SASTORE")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        ASTORE2helper();
    }
    public void visitPOP() {
        super.visitPOP();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": POP")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Reg_Mem(x86.LEA, ESP, 4, ESP);
    }
    public void visitPOP2() {
        super.visitPOP2();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": POP2")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Reg_Mem(x86.LEA, ESP, 8, ESP);
    }
    public void visitDUP() {
        super.visitDUP();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": DUP")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Mem(x86.PUSH_m, 0, ESP);
    }
    public void visitDUP_x1() {
        super.visitDUP_x1();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": DUP_x1")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitShort_Reg(x86.POP_r, EAX);
        asm.emitShort_Reg(x86.POP_r, EBX);
        asm.emitShort_Reg(x86.PUSH_r, EAX);
        asm.emitShort_Reg(x86.PUSH_r, EBX);
        asm.emitShort_Reg(x86.PUSH_r, EAX);
    }
    public void visitDUP_x2() {
        super.visitDUP_x2();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": DUP_x2")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitShort_Reg(x86.POP_r, EAX);
        asm.emitShort_Reg(x86.POP_r, EBX);
        asm.emitShort_Reg(x86.POP_r, ECX);
        asm.emitShort_Reg(x86.PUSH_r, EAX);
        asm.emitShort_Reg(x86.PUSH_r, ECX);
        asm.emitShort_Reg(x86.PUSH_r, EBX);
        asm.emitShort_Reg(x86.PUSH_r, EAX);
    }
    public void visitDUP2() {
        super.visitDUP2();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": DUP2")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitShort_Reg(x86.POP_r, EAX);
        asm.emitShort_Reg(x86.POP_r, EBX);
        asm.emitShort_Reg(x86.PUSH_r, EBX);
        asm.emitShort_Reg(x86.PUSH_r, EAX);
        asm.emitShort_Reg(x86.PUSH_r, EBX);
        asm.emitShort_Reg(x86.PUSH_r, EAX);
    }
    public void visitDUP2_x1() {
        super.visitDUP2_x1();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": DUP2_x1")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitShort_Reg(x86.POP_r, EAX);
        asm.emitShort_Reg(x86.POP_r, EBX);
        asm.emitShort_Reg(x86.POP_r, ECX);
        asm.emitShort_Reg(x86.PUSH_r, EBX);
        asm.emitShort_Reg(x86.PUSH_r, EAX);
        asm.emitShort_Reg(x86.PUSH_r, ECX);
        asm.emitShort_Reg(x86.PUSH_r, EBX);
        asm.emitShort_Reg(x86.PUSH_r, EAX);
    }
    public void visitDUP2_x2() {
        super.visitDUP2_x2();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": DUP2_x2")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitShort_Reg(x86.POP_r, EAX);
        asm.emitShort_Reg(x86.POP_r, EBX);
        asm.emitShort_Reg(x86.POP_r, ECX);
        asm.emitShort_Reg(x86.POP_r, EDX);
        asm.emitShort_Reg(x86.PUSH_r, EBX);
        asm.emitShort_Reg(x86.PUSH_r, EAX);
        asm.emitShort_Reg(x86.PUSH_r, EDX);
        asm.emitShort_Reg(x86.PUSH_r, ECX);
        asm.emitShort_Reg(x86.PUSH_r, EBX);
        asm.emitShort_Reg(x86.PUSH_r, EAX);
    }
    public void visitSWAP() {
        super.visitSWAP();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": SWAP")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitShort_Reg(x86.POP_r, EAX);
        asm.emitShort_Reg(x86.POP_r, EBX);
        asm.emitShort_Reg(x86.PUSH_r, EAX);
        asm.emitShort_Reg(x86.PUSH_r, EBX);
    }
    public void visitIBINOP(byte op) {
        super.visitIBINOP(op);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": IBINOP "+op)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        switch(op) {
            case BINOP_ADD:
                asm.emitShort_Reg(x86.POP_r, EAX);
                asm.emitARITH_Reg_Mem(x86.ADD_m_r32, EAX, 0, ESP);
                break;
            case BINOP_SUB:
                asm.emitShort_Reg(x86.POP_r, EAX);
                asm.emitARITH_Reg_Mem(x86.SUB_m_r32, EAX, 0, ESP); // a-b
                break;
            case BINOP_MUL:
                asm.emitShort_Reg(x86.POP_r, EAX);
                asm.emit2_Mem(x86.IMUL_rda_m32, 0, ESP);
                asm.emit2_Reg_Mem(x86.MOV_m_r32, EAX, 0, ESP);
                break;
            case BINOP_DIV:
                asm.emitShort_Reg(x86.POP_r, ECX);
                asm.emitShort_Reg(x86.POP_r, EAX);
                asm.emit1(x86.CWD);
                asm.emit2_Reg(x86.IDIV_r32, ECX);
                asm.emitShort_Reg(x86.PUSH_r, EAX);
                break;
            case BINOP_REM:
                asm.emitShort_Reg(x86.POP_r, ECX);
                asm.emitShort_Reg(x86.POP_r, EAX);
                asm.emit1(x86.CWD);
                asm.emit2_Reg(x86.IDIV_r32, ECX);
                asm.emitShort_Reg(x86.PUSH_r, EDX);
                break;
            case BINOP_AND:
                asm.emitShort_Reg(x86.POP_r, EAX);
                asm.emit2_Reg_Mem(x86.AND_m_r32, EAX, 0, ESP);
                break;
            case BINOP_OR:
                asm.emitShort_Reg(x86.POP_r, EAX);
                asm.emit2_Reg_Mem(x86.OR_m_r32, EAX, 0, ESP);
                break;
            case BINOP_XOR:
                asm.emitShort_Reg(x86.POP_r, EAX);
                asm.emit2_Reg_Mem(x86.XOR_m_r32, EAX, 0, ESP);
                break;
            default:
                jq.UNREACHABLE();
        }
    }
    public void visitLBINOP(byte op) {
        super.visitLBINOP(op);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": LBINOP "+op)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        switch(op) {
            case BINOP_ADD:
                asm.emitShort_Reg(x86.POP_r, EAX); // lo
                asm.emitShort_Reg(x86.POP_r, EBX); // hi
                asm.emit2_Reg_Mem(x86.ADD_m_r32, EAX, 0, ESP);
                asm.emit2_Reg_Mem(x86.ADC_m_r32, EBX, 4, ESP);
                break;
            case BINOP_SUB:
                asm.emitShort_Reg(x86.POP_r, EAX); // lo
                asm.emitShort_Reg(x86.POP_r, EBX); // hi
                asm.emit2_Reg_Mem(x86.SUB_m_r32, EAX, 0, ESP);
                asm.emit2_Reg_Mem(x86.SBB_m_r32, EBX, 4, ESP);
                break;
            case BINOP_MUL:
                asm.emitShort_Reg(x86.POP_r, EBX); // lo1
                asm.emitShort_Reg(x86.POP_r, ECX); // hi1
                asm.emitShort_Reg(x86.POP_r, ESI); // lo2
                asm.emitShort_Reg(x86.POP_r, EDI); // hi2
                asm.emit2_Reg_Reg(x86.MOV_r_r32, EAX, EDI); // hi2
                asm.emitARITH_Reg_Reg(x86.OR_r_r32, EAX, ECX); // hi1 | hi2
                asm.emitCJUMP_Short(x86.JNE, (byte)0);
                int cloc = asm.getCurrentOffset();
                asm.emit2_Reg_Reg(x86.MOV_r_r32, EAX, ESI); // lo2
                asm.emit2_Reg(x86.MUL_rda_r32, EBX); // lo1*lo2
                asm.emitJUMP_Short(x86.JMP, (byte)0);
                asm.patch1(cloc-1, (byte)(asm.getCurrentOffset()-cloc));
                cloc = asm.getCurrentOffset();
                asm.emit2_Reg_Reg(x86.MOV_r_r32, EAX, ESI); // lo2
                asm.emit2_Reg(x86.MUL_rda_r32, ECX); // hi1*lo2
                asm.emit2_Reg_Reg(x86.MOV_r_r32, ECX, EAX); // hi1*lo2
                asm.emit2_Reg_Reg(x86.MOV_r_r32, EAX, EDI); // hi2
                asm.emit2_Reg(x86.MUL_rda_r32, EBX); // hi2*lo1
                asm.emitARITH_Reg_Reg(x86.ADD_r_r32, ECX, EAX); // hi2*lo1 + hi1*lo2
                asm.emit2_Reg_Reg(x86.MOV_r_r32, EAX, ESI); // lo2
                asm.emit2_Reg(x86.MUL_rda_r32, EBX); // lo1*lo2
                asm.emitARITH_Reg_Reg(x86.ADD_r_r32, EDX, ECX); // hi2*lo1 + hi1*lo2 + hi(lo1*lo2)
                asm.patch1(cloc-1, (byte)(asm.getCurrentOffset()-cloc));
                asm.emitShort_Reg(x86.PUSH_r, EDX); // res_hi
                asm.emitShort_Reg(x86.PUSH_r, EAX); // res_lo
                break;
            case BINOP_DIV:
                asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(MathSupport._ldiv));
                asm.emitShort_Reg(x86.PUSH_r, EDX); // hi
                asm.emitShort_Reg(x86.PUSH_r, EAX); // lo
                break;
            case BINOP_REM:
                asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(MathSupport._lrem));
                asm.emitShort_Reg(x86.PUSH_r, EDX); // hi
                asm.emitShort_Reg(x86.PUSH_r, EAX); // lo
                break;
            case BINOP_AND:
                asm.emitShort_Reg(x86.POP_r, EAX); // lo
                asm.emitShort_Reg(x86.POP_r, EDX); // hi
                asm.emit2_Reg_Mem(x86.AND_m_r32, EAX, 0, ESP);
                asm.emit2_Reg_Mem(x86.AND_m_r32, EDX, 4, ESP);
                break;
            case BINOP_OR:
                asm.emitShort_Reg(x86.POP_r, EAX); // lo
                asm.emitShort_Reg(x86.POP_r, EDX); // hi
                asm.emit2_Reg_Mem(x86.OR_m_r32, EAX, 0, ESP);
                asm.emit2_Reg_Mem(x86.OR_m_r32, EDX, 4, ESP);
                break;
            case BINOP_XOR:
                asm.emitShort_Reg(x86.POP_r, EAX); // lo
                asm.emitShort_Reg(x86.POP_r, EDX); // hi
                asm.emit2_Reg_Mem(x86.XOR_m_r32, EAX, 0, ESP);
                asm.emit2_Reg_Mem(x86.XOR_m_r32, EDX, 4, ESP);
                break;
            default:
                jq.UNREACHABLE();
        }
    }
    public void visitFBINOP(byte op) {
        super.visitFBINOP(op);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": FBINOP "+op)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        switch(op) {
            case BINOP_ADD:
                asm.emit2_Mem(x86.FLD_m32, 4, ESP);
                asm.emit2_Mem(x86.FADD_m32, 0, ESP);
                asm.emit2_Reg_Mem(x86.LEA, ESP, 4, ESP);
                asm.emit2_Mem(x86.FSTP_m32, 0, ESP);
                break;
            case BINOP_SUB:
                asm.emit2_Mem(x86.FLD_m32, 4, ESP);
                asm.emit2_Mem(x86.FSUB_m32, 0, ESP);
                asm.emit2_Reg_Mem(x86.LEA, ESP, 4, ESP);
                asm.emit2_Mem(x86.FSTP_m32, 0, ESP);
                break;
            case BINOP_MUL:
                asm.emit2_Mem(x86.FLD_m32, 4, ESP);
                asm.emit2_Mem(x86.FMUL_m32, 0, ESP);
                asm.emit2_Reg_Mem(x86.LEA, ESP, 4, ESP);
                asm.emit2_Mem(x86.FSTP_m32, 0, ESP);
                break;
            case BINOP_DIV:
                asm.emit2_Mem(x86.FLD_m32, 4, ESP);
                asm.emit2_Mem(x86.FDIV_m32, 0, ESP);
                asm.emit2_Reg_Mem(x86.LEA, ESP, 4, ESP);
                asm.emit2_Mem(x86.FSTP_m32, 0, ESP);
                break;
            case BINOP_REM:
                asm.emit2_Mem(x86.FLD_m32, 0, ESP); // reverse because pushing on fp stack
                asm.emit2_Mem(x86.FLD_m32, 4, ESP);
                asm.emit2(x86.FPREM);
                asm.emit2_Reg_Mem(x86.LEA, ESP, 4, ESP);
                asm.emit2_Mem(x86.FSTP_m32, 0, ESP);
                asm.emit2_FPReg(x86.FFREE, 0);
                break;
            default:
                jq.UNREACHABLE();
        }
    }
    public void visitDBINOP(byte op) {
        super.visitDBINOP(op);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": DBINOP "+op)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        switch(op) {
            case BINOP_ADD:
                asm.emit2_Mem(x86.FLD_m64, 8, ESP);
                asm.emit2_Mem(x86.FADD_m64, 0, ESP);
                asm.emit2_Reg_Mem(x86.LEA, ESP, 8, ESP);
                asm.emit2_Mem(x86.FSTP_m64, 0, ESP);
                break;
            case BINOP_SUB:
                asm.emit2_Mem(x86.FLD_m64, 8, ESP);
                asm.emit2_Mem(x86.FSUB_m64, 0, ESP);
                asm.emit2_Reg_Mem(x86.LEA, ESP, 8, ESP);
                asm.emit2_Mem(x86.FSTP_m64, 0, ESP);
                break;
            case BINOP_MUL:
                asm.emit2_Mem(x86.FLD_m64, 8, ESP);
                asm.emit2_Mem(x86.FMUL_m64, 0, ESP);
                asm.emit2_Reg_Mem(x86.LEA, ESP, 8, ESP);
                asm.emit2_Mem(x86.FSTP_m64, 0, ESP);
                break;
            case BINOP_DIV:
                asm.emit2_Mem(x86.FLD_m64, 8, ESP);
                asm.emit2_Mem(x86.FDIV_m64, 0, ESP);
                asm.emit2_Reg_Mem(x86.LEA, ESP, 8, ESP);
                asm.emit2_Mem(x86.FSTP_m64, 0, ESP);
                break;
            case BINOP_REM:
                asm.emit2_Mem(x86.FLD_m64, 0, ESP);
                asm.emit2_Mem(x86.FLD_m64, 8, ESP);
                asm.emit2(x86.FPREM);
                asm.emit2_Reg_Mem(x86.LEA, ESP, 8, ESP);
                asm.emit2_Mem(x86.FSTP_m64, 0, ESP);
                asm.emit2_FPReg(x86.FFREE, 0);
                break;
            default:
                jq.UNREACHABLE();
        }
    }
    public void visitIUNOP(byte op) {
        super.visitIUNOP(op);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": IUNOP "+op)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        jq.assert(op == UNOP_NEG);
        asm.emit2_Mem(x86.NEG_m32, 0, ESP);
    }
    public void visitLUNOP(byte op) {
        super.visitLUNOP(op);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": LUNOP "+op)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        jq.assert(op == UNOP_NEG);
        asm.emit2_Mem(x86.NEG_m32, 4, ESP);  // hi
        asm.emit2_Mem(x86.NEG_m32, 0, ESP);  // lo
        asm.emitARITH_Mem_Imm(x86.SBB_m_i32, 4, ESP, 0);
    }
    public void visitFUNOP(byte op) {
        super.visitFUNOP(op);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": FUNOP "+op)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        jq.assert(op == UNOP_NEG);
        asm.emit2_Mem(x86.FLD_m32, 0, ESP);
        asm.emit2(x86.FCHS);
        asm.emit2_Mem(x86.FSTP_m32, 0, ESP);
    }
    public void visitDUNOP(byte op) {
        super.visitDUNOP(op);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": DUNOP "+op)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Mem(x86.FLD_m64, 0, ESP);
        asm.emit2(x86.FCHS);
        asm.emit2_Mem(x86.FSTP_m64, 0, ESP);
    }
    public void visitISHIFT(byte op) {
        super.visitISHIFT(op);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": ISHIFT "+op)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        switch(op) {
            case SHIFT_LEFT:
                asm.emitShort_Reg(x86.POP_r, ECX);
                asm.emit2_Mem(x86.SHL_m32_rc, 0, ESP);
                break;
            case SHIFT_RIGHT:
                asm.emitShort_Reg(x86.POP_r, ECX);
                asm.emit2_Mem(x86.SAR_m32_rc, 0, ESP);
                break;
            case SHIFT_URIGHT:
                asm.emitShort_Reg(x86.POP_r, ECX);
                asm.emit2_Mem(x86.SHR_m32_rc, 0, ESP);
                break;
            default:
                jq.UNREACHABLE();
        }
    }
    public void visitLSHIFT(byte op) {
        super.visitLSHIFT(op);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": LSHIFT"+op)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        switch(op) {
            case SHIFT_LEFT: {
                asm.emitShort_Reg(x86.POP_r, ECX);
                asm.emitShort_Reg(x86.POP_r, EAX); // lo
                asm.emitShort_Reg(x86.POP_r, EDX); // hi
                asm.emitARITH_Reg_Imm(x86.AND_r_i32, ECX, 63);
                asm.emitARITH_Reg_Imm(x86.CMP_r_i32, ECX, 32);
                asm.emitCJUMP_Short(x86.JAE, (byte)0);
                int cloc = asm.getCurrentOffset();
                asm.emitSHLD_r_r_rc(EDX, EAX);
                asm.emit2_Reg(x86.SHL_r32_rc, EAX);
                asm.emitJUMP_Short(x86.JMP, (byte)0);
                asm.patch1(cloc-1, (byte)(asm.getCurrentOffset()-cloc));
                cloc = asm.getCurrentOffset();
                asm.emit2_Reg_Reg(x86.MOV_r_r32, EDX, EAX);
                asm.emitARITH_Reg_Reg(x86.XOR_r_r32, EAX, EAX);
                asm.emit2_Reg(x86.SHL_r32_rc, EDX);
                asm.patch1(cloc-1, (byte)(asm.getCurrentOffset()-cloc));
                asm.emitShort_Reg(x86.PUSH_r, EDX); // hi
                asm.emitShort_Reg(x86.PUSH_r, EAX); // lo
                break;
            }
            case SHIFT_RIGHT: {
                asm.emitShort_Reg(x86.POP_r, ECX);
                asm.emitShort_Reg(x86.POP_r, EAX); // lo
                asm.emitShort_Reg(x86.POP_r, EDX); // hi
                asm.emitARITH_Reg_Imm(x86.AND_r_i32, ECX, 63);
                asm.emitARITH_Reg_Imm(x86.CMP_r_i32, ECX, 32);
                asm.emitCJUMP_Short(x86.JAE, (byte)0);
                int cloc = asm.getCurrentOffset();
                asm.emitSHRD_r_r_rc(EAX, EDX);
                asm.emit2_Reg(x86.SAR_r32_rc, EDX);
                asm.emitJUMP_Short(x86.JMP, (byte)0);
                asm.patch1(cloc-1, (byte)(asm.getCurrentOffset()-cloc));
                cloc = asm.getCurrentOffset();
                asm.emit2_Reg_Reg(x86.MOV_r_r32, EAX, EDX);
                asm.emit2_SHIFT_Reg_Imm8(x86.SAR_r32_i, EDX, (byte)31);
                asm.emit2_Reg(x86.SAR_r32_rc, EAX);
                asm.patch1(cloc-1, (byte)(asm.getCurrentOffset()-cloc));
                asm.emitShort_Reg(x86.PUSH_r, EDX); // hi
                asm.emitShort_Reg(x86.PUSH_r, EAX); // lo
                break;
            }
            case SHIFT_URIGHT: {
                asm.emitShort_Reg(x86.POP_r, ECX);
                asm.emitShort_Reg(x86.POP_r, EAX); // lo
                asm.emitShort_Reg(x86.POP_r, EDX); // hi
                asm.emitARITH_Reg_Imm(x86.AND_r_i32, ECX, 63);
                asm.emitARITH_Reg_Imm(x86.CMP_r_i32, ECX, 32);
                asm.emitCJUMP_Short(x86.JAE, (byte)0);
                int cloc = asm.getCurrentOffset();
                asm.emitSHRD_r_r_rc(EAX, EDX);
                asm.emit2_Reg(x86.SHR_r32_rc, EDX);
                asm.emitJUMP_Short(x86.JMP, (byte)0);
                asm.patch1(cloc-1, (byte)(asm.getCurrentOffset()-cloc));
                cloc = asm.getCurrentOffset();
                asm.emit2_Reg_Reg(x86.MOV_r_r32, EAX, EDX);
                asm.emitARITH_Reg_Reg(x86.XOR_r_r32, EDX, EDX);
                asm.emit2_Reg(x86.SHR_r32_rc, EAX);
                asm.patch1(cloc-1, (byte)(asm.getCurrentOffset()-cloc));
                asm.emitShort_Reg(x86.PUSH_r, EDX); // hi
                asm.emitShort_Reg(x86.PUSH_r, EAX); // lo
                break;
            }
            default:
                jq.UNREACHABLE();
        }
    }
    public void visitIINC(int i, int v) {
        super.visitIINC(i, v);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": IINC "+i+" "+v)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitARITH_Mem_Imm(x86.ADD_m_i32, getLocalOffset(i), EBP, v);
    }
    public void visitI2L() {
        super.visitI2L();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": I2L")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitShort_Reg(x86.POP_r, EAX); // lo
        asm.emit1(x86.CWD);
        asm.emitShort_Reg(x86.PUSH_r, EDX); // hi
        asm.emitShort_Reg(x86.PUSH_r, EAX); // lo
    }
    public void visitI2F() {
        super.visitI2F();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": I2F")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Mem(x86.FILD_m32, 0, ESP);
        asm.emit2_Mem(x86.FSTP_m32, 0, ESP);
    }
    public void visitI2D() {
        super.visitI2D();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": I2D")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Mem(x86.FILD_m32, 0, ESP);
        asm.emit2_Reg_Mem(x86.LEA, ESP, -4, ESP);
        asm.emit2_Mem(x86.FSTP_m64, 0, ESP);
    }
    public void visitL2I() {
        super.visitL2I();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": L2I")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitShort_Reg(x86.POP_r, EAX); // lo
        asm.emitShort_Reg(x86.POP_r, ECX); // hi
        asm.emitShort_Reg(x86.PUSH_r, EAX);
    }
    public void visitL2F() {
        super.visitL2F();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": L2F")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Mem(x86.FILD_m64, 0, ESP);
        asm.emit2_Reg_Mem(x86.LEA, ESP, 4, ESP);
        asm.emit2_Mem(x86.FSTP_m32, 0, ESP);
    }
    public void visitL2D() {
        super.visitL2D();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": L2D")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Mem(x86.FILD_m64, 0, ESP);
        asm.emit2_Mem(x86.FSTP_m64, 0, ESP);
    }
    private void toIntHelper() {
        // check for NaN
        asm.emit2_Mem(x86.FLD_m64, MathSupport._maxint.getAddress());
        asm.emit2_FPReg(x86.FUCOMIP, 1);
        asm.emitCJUMP_Short(x86.JP, (byte)0);
        int cloc1 = asm.getCurrentOffset();
        // check for >=MAX_INT
        asm.emitCJUMP_Short(x86.JBE, (byte)0);
        int cloc2 = asm.getCurrentOffset();
        // check for <=MIN_INT
        asm.emit2_Mem(x86.FLD_m64, MathSupport._minint.getAddress());
        asm.emit2_FPReg(x86.FUCOMIP, 1);
        asm.emitCJUMP_Short(x86.JAE, (byte)0);
        int cloc3 = asm.getCurrentOffset();
        // default case
        {   // set rounding mode to round-towards-zero
            asm.emit2_Mem(x86.FNSTCW, -4, ESP);
            asm.emit2_Mem(x86.FNSTCW, -8, ESP);
            asm.emitARITH_Mem_Imm(x86.OR_m_i32, -4, ESP, 0x0c00);
            asm.emit2(x86.FNCLEX);
            asm.emit2_Mem(x86.FLDCW, -4, ESP);
        }
        asm.emit2_Mem(x86.FISTP_m32, 0, ESP);
        {
            // restore fpu control word
            asm.emit2(x86.FNCLEX);
            asm.emit2_Mem(x86.FLDCW, -8, ESP);
        }
        asm.emitJUMP_Short(x86.JMP, (byte)0);
        int cloc4 = asm.getCurrentOffset();
        asm.patch1(cloc1-1, (byte)(asm.getCurrentOffset()-cloc1));
        // NaN -> 0
        asm.emit2_Mem_Imm(x86.MOV_m_i32, 0, ESP, 0);
        asm.emitJUMP_Short(x86.JMP, (byte)0);
        int cloc5 = asm.getCurrentOffset();
        asm.patch1(cloc2-1, (byte)(asm.getCurrentOffset()-cloc2));
        // >=MAX_INT -> MAX_INT
        asm.emit2_Mem_Imm(x86.MOV_m_i32, 0, ESP, Integer.MAX_VALUE);
        asm.emitJUMP_Short(x86.JMP, (byte)0);
        int cloc6 = asm.getCurrentOffset();
        asm.patch1(cloc3-1, (byte)(asm.getCurrentOffset()-cloc3));
        // <=MIN_INT -> MIN_INT
        asm.emit2_Mem_Imm(x86.MOV_m_i32, 0, ESP, Integer.MIN_VALUE);
        asm.patch1(cloc5-1, (byte)(asm.getCurrentOffset()-cloc5));
        asm.patch1(cloc6-1, (byte)(asm.getCurrentOffset()-cloc6));
        asm.emit2_FPReg(x86.FFREE, 0);
        asm.patch1(cloc4-1, (byte)(asm.getCurrentOffset()-cloc4));
    }
    private void toLongHelper() {
        // check for NaN
        asm.emit2_Mem(x86.FLD_m64, MathSupport._maxlong.getAddress());
        asm.emit2_FPReg(x86.FUCOMIP, 1);
        asm.emitCJUMP_Short(x86.JP, (byte)0);
        int cloc1 = asm.getCurrentOffset();
        // check for >=MAX_LONG
        asm.emitCJUMP_Short(x86.JBE, (byte)0);
        int cloc2 = asm.getCurrentOffset();
        // check for <=MIN_LONG
        asm.emit2_Mem(x86.FLD_m64, MathSupport._minlong.getAddress());
        asm.emit2_FPReg(x86.FUCOMIP, 1);
        asm.emitCJUMP_Short(x86.JAE, (byte)0);
        int cloc3 = asm.getCurrentOffset();
        // default case
        {   // set rounding mode to round-towards-zero
            asm.emit2_Mem(x86.FNSTCW, -4, ESP);
            asm.emit2_Mem(x86.FNSTCW, -8, ESP);
            asm.emitARITH_Mem_Imm(x86.OR_m_i32, -4, ESP, 0x0c00);
            asm.emit2(x86.FNCLEX);
            asm.emit2_Mem(x86.FLDCW, -4, ESP);
        }
        asm.emit2_Mem(x86.FISTP_m64, 0, ESP);
        {
            // restore fpu control word
            asm.emit2(x86.FNCLEX);
            asm.emit2_Mem(x86.FLDCW, -8, ESP);
        }
        asm.emitJUMP_Short(x86.JMP, (byte)0);
        int cloc4 = asm.getCurrentOffset();
        asm.patch1(cloc1-1, (byte)(asm.getCurrentOffset()-cloc1));
        // NaN -> 0
        asm.emit2_Mem_Imm(x86.MOV_m_i32, 0, ESP, 0);
        asm.emit2_Mem_Imm(x86.MOV_m_i32, 4, ESP, 0);
        asm.emitJUMP_Short(x86.JMP, (byte)0);
        int cloc5 = asm.getCurrentOffset();
        asm.patch1(cloc2-1, (byte)(asm.getCurrentOffset()-cloc2));
        // >=MAX_LONG -> MAX_LONG
        asm.emit2_Mem_Imm(x86.MOV_m_i32, 0, ESP, (int)Long.MAX_VALUE);
        asm.emit2_Mem_Imm(x86.MOV_m_i32, 4, ESP, (int)(Long.MAX_VALUE>>32));
        asm.emitJUMP_Short(x86.JMP, (byte)0);
        int cloc6 = asm.getCurrentOffset();
        asm.patch1(cloc3-1, (byte)(asm.getCurrentOffset()-cloc3));
        // <=MIN_LONG -> MIN_LONG
        asm.emit2_Mem_Imm(x86.MOV_m_i32, 0, ESP, (int)Long.MIN_VALUE);
        asm.emit2_Mem_Imm(x86.MOV_m_i32, 4, ESP, (int)(Long.MIN_VALUE>>32));
        asm.patch1(cloc5-1, (byte)(asm.getCurrentOffset()-cloc5));
        asm.patch1(cloc6-1, (byte)(asm.getCurrentOffset()-cloc6));
        asm.emit2_FPReg(x86.FFREE, 0);
        asm.patch1(cloc4-1, (byte)(asm.getCurrentOffset()-cloc4));
    }
    public void visitF2I() {
        super.visitF2I();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": F2I")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Mem(x86.FLD_m32, 0, ESP);
        toIntHelper();
    }
    public void visitF2L() {
        super.visitF2L();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": F2L")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Mem(x86.FLD_m32, 0, ESP);
        asm.emit2_Reg_Mem(x86.LEA, ESP, -4, ESP);
        toLongHelper();
    }
    public void visitF2D() {
        super.visitF2D();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": F2D")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Mem(x86.FLD_m32, 0, ESP);
        asm.emit2_Reg_Mem(x86.LEA, ESP, -4, ESP);
        asm.emit2_Mem(x86.FSTP_m64, 0, ESP);
    }
    public void visitD2I() {
        super.visitD2I();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": D2I")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Mem(x86.FLD_m64, 0, ESP);
        asm.emit2_Reg_Mem(x86.LEA, ESP, 4, ESP);
        toIntHelper();
    }
    public void visitD2L() {
        super.visitD2L();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": D2L")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Mem(x86.FLD_m64, 0, ESP);
        toLongHelper();
    }
    public void visitD2F() {
        super.visitD2F();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": D2F")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Mem(x86.FLD_m64, 0, ESP);
        asm.emit2_Reg_Mem(x86.LEA, ESP, 4, ESP);
        asm.emit2_Mem(x86.FSTP_m32, 0, ESP);
    }
    public void visitI2B() {
        super.visitI2B();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": I2B")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitShort_Reg(x86.POP_r, EAX);
        asm.emit3_Reg_Reg(x86.MOVSX_r_r8, EAX, AL);
        asm.emitShort_Reg(x86.PUSH_r, EAX);
    }
    public void visitI2C() {
        super.visitI2C();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": I2C")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitShort_Reg(x86.POP_r, EAX);
        asm.emit3_Reg_Reg(x86.MOVZX_r_r16, EAX, AX);
        asm.emitShort_Reg(x86.PUSH_r, EAX);
    }
    public void visitI2S() {
        super.visitI2S();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": I2S")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitShort_Reg(x86.POP_r, EAX);
        asm.emit3_Reg_Reg(x86.MOVSX_r_r16, EAX, AX);
        asm.emitShort_Reg(x86.PUSH_r, EAX);
    }
    public void visitLCMP2() {
        super.visitLCMP2();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": LCMP2")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitShort_Reg(x86.POP_r, EBX); // lo
        asm.emitShort_Reg(x86.POP_r, ECX); // hi
        asm.emitShort_Reg(x86.POP_r, EAX); // lo
        asm.emitShort_Reg(x86.POP_r, EDX); // hi
        asm.emitARITH_Reg_Reg(x86.SUB_r_r32, EAX, EBX);
        asm.emitARITH_Reg_Reg(x86.SBB_r_r32, EDX, ECX);
        asm.emitShort_Reg_Imm(x86.MOV_r_i32, ECX, -1);
        asm.emitCJUMP_Short(x86.JL, (byte)0);
        int cloc1 = asm.getCurrentOffset();
        asm.emitARITH_Reg_Reg(x86.XOR_r_r32, ECX, ECX);
        asm.emitARITH_Reg_Reg(x86.OR_r_r32, EAX, EDX);
        asm.emitCJUMP_Short(x86.JE, (byte)0);
        int cloc2 = asm.getCurrentOffset();
        asm.emitShort_Reg(x86.INC_r32, ECX);
        asm.patch1(cloc1-1, (byte)(asm.getCurrentOffset()-cloc1));
        asm.patch1(cloc2-1, (byte)(asm.getCurrentOffset()-cloc2));
        asm.emitShort_Reg(x86.PUSH_r, ECX);
    }
    public void visitFCMP2(byte op) {
        super.visitFCMP2(op);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": FCMP2 "+op)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        if (op == CMP_L) {
            asm.emit2_Mem(x86.FLD_m32, 0, ESP);
            asm.emit2_Mem(x86.FLD_m32, 4, ESP);
        } else {
            asm.emit2_Mem(x86.FLD_m32, 4, ESP); // reverse order
            asm.emit2_Mem(x86.FLD_m32, 0, ESP);
        }
        asm.emit2(x86.FUCOMPP);
        asm.emit2(x86.FNSTSW_ax);
        asm.emit1(x86.SAHF);
        asm.emit2_Reg_Mem(x86.LEA, ESP, 8, ESP);
        if (op == CMP_L) {
            asm.emitShort_Reg_Imm(x86.MOV_r_i32, ECX, -1);
        } else {
            asm.emitShort_Reg_Imm(x86.MOV_r_i32, ECX, 1);
        }
        asm.emitCJUMP_Short(x86.JB, (byte)0);
        int cloc1 = asm.getCurrentOffset();
        asm.emitShort_Reg_Imm(x86.MOV_r_i32, ECX, 0);
        asm.emitCJUMP_Short(x86.JE, (byte)0);
        int cloc2 = asm.getCurrentOffset();
        if (op == CMP_L) {
            asm.emitShort_Reg(x86.INC_r32, ECX);
        } else {
            asm.emitShort_Reg(x86.DEC_r32, ECX);
        }
        asm.patch1(cloc1-1, (byte)(asm.getCurrentOffset()-cloc1));
        asm.patch1(cloc2-1, (byte)(asm.getCurrentOffset()-cloc2));
        asm.emitShort_Reg(x86.PUSH_r, ECX);
    }
    public void visitDCMP2(byte op) {
        super.visitDCMP2(op);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": DCMP2 "+op)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        if (op == CMP_L) {
            asm.emit2_Mem(x86.FLD_m64, 0, ESP);
            asm.emit2_Mem(x86.FLD_m64, 8, ESP);
        } else {
            asm.emit2_Mem(x86.FLD_m64, 8, ESP); // reverse order
            asm.emit2_Mem(x86.FLD_m64, 0, ESP);
        }
        asm.emit2(x86.FUCOMPP);
        asm.emit2(x86.FNSTSW_ax);
        asm.emit1(x86.SAHF);
        asm.emit2_Reg_Mem(x86.LEA, ESP, 16, ESP);
        if (op == CMP_L) {
            asm.emitShort_Reg_Imm(x86.MOV_r_i32, ECX, -1);
        } else {
            asm.emitShort_Reg_Imm(x86.MOV_r_i32, ECX, 1);
        }
        asm.emitCJUMP_Short(x86.JB, (byte)0);
        int cloc1 = asm.getCurrentOffset();
        asm.emitShort_Reg_Imm(x86.MOV_r_i32, ECX, 0);
        asm.emitCJUMP_Short(x86.JE, (byte)0);
        int cloc2 = asm.getCurrentOffset();
        if (op == CMP_L) {
            asm.emitShort_Reg(x86.INC_r32, ECX);
        } else {
            asm.emitShort_Reg(x86.DEC_r32, ECX);
        }
        asm.patch1(cloc1-1, (byte)(asm.getCurrentOffset()-cloc1));
        asm.patch1(cloc2-1, (byte)(asm.getCurrentOffset()-cloc2));
        asm.emitShort_Reg(x86.PUSH_r, ECX);
    }
    private void branchHelper(byte op, int target) {
        Integer t = new Integer(target);
        if (op == CMP_UNCOND)
            if (target <= i_start)
                asm.emitJUMP_Back(x86.JMP, t);
            else
                asm.emitJUMP_Forw(x86.JMP, t);
        else {
            x86 opc = null;
            switch(op) {
                case CMP_EQ: opc = x86.JE; break;
                case CMP_NE: opc = x86.JNE; break;
                case CMP_LT: opc = x86.JL; break;
                case CMP_GE: opc = x86.JGE; break;
                case CMP_LE: opc = x86.JLE; break;
                case CMP_GT: opc = x86.JG; break;
                case CMP_AE: opc = x86.JAE; break;
                default: jq.UNREACHABLE();
            }
            if (target <= i_start)
                asm.emitCJUMP_Back(opc, t);
            else
                asm.emitCJUMP_Forw(opc, t);
        }
    }
    public void visitIF(byte op, int target) {
        super.visitIF(op, target);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": IF "+op+" "+target)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitShort_Reg(x86.POP_r, EAX);
        asm.emitARITH_Reg_Imm(x86.CMP_r_i32, EAX, 0);
        branchHelper(op, target);
    }
    public void visitIFREF(byte op, int target) {
        super.visitIFREF(op, target);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": IFREF "+op+" "+target)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitShort_Reg(x86.POP_r, EAX);
        asm.emitARITH_Reg_Imm(x86.CMP_r_i32, EAX, 0);
        branchHelper(op, target);
    }
    public void visitIFCMP(byte op, int target) {
        super.visitIFCMP(op, target);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": IFCMP "+op+" "+target)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitShort_Reg(x86.POP_r, ECX);
        asm.emitShort_Reg(x86.POP_r, EAX);
        asm.emitARITH_Reg_Reg(x86.CMP_r_r32, EAX, ECX);
        branchHelper(op, target);
    }
    public void visitIFREFCMP(byte op, int target) {
        super.visitIFREFCMP(op, target);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": IFREFCMP "+op+" "+target)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitShort_Reg(x86.POP_r, ECX);
        asm.emitShort_Reg(x86.POP_r, EAX);
        asm.emitARITH_Reg_Reg(x86.CMP_r_r32, EAX, ECX);
        branchHelper(op, target);
    }
    public void visitGOTO(int target) {
        super.visitGOTO(target);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": GOTO "+target)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        branchHelper(CMP_UNCOND, target);
    }
    public void visitJSR(int target) {
        super.visitJSR(target);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": JSR "+target)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        Integer t = new Integer(target);
        if (target <= i_start) {
            asm.emitCALL_Back(x86.CALL_rel32, t);
        } else {
            asm.emitCALL_Forw(x86.CALL_rel32, t);
        }
    }
    public void visitRET(int i) {
        super.visitRET(i);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": RET "+i)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit2_Mem(x86.JMP_m, getLocalOffset(i), EBP);
    }
    public void visitTABLESWITCH(int default_target, int low, int high, int[] targets) {
        super.visitTABLESWITCH(default_target, low, high, targets);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": TABLESWITCH "+default_target+" "+low+" "+high)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        int count = high-low+1;
        jq.assert(count == targets.length);
        asm.emitShort_Reg(x86.POP_r, EAX);
        if (low != 0)
            asm.emitARITH_Reg_Imm(x86.SUB_r_i32, EAX, low);
        asm.emitARITH_Reg_Imm(x86.CMP_r_i32, EAX, count);
        branchHelper(CMP_AE, default_target);
        asm.emitCALL_rel(x86.CALL_rel32, 0);
        int cloc = asm.getCurrentOffset();
        asm.emitShort_Reg(x86.POP_r, ECX);
        // val from table + abs position in table
        asm.emit2_Reg_Mem(x86.LEA, EDX, ECX, EAX, SCALE_4, 127);
        int cloc2 = asm.getCurrentOffset();
        asm.emitARITH_Reg_Mem(x86.ADD_r_m32, EDX, -4, EDX);
        asm.emit2_Reg(x86.JMP_r, EDX);
        asm.patch1(cloc2-1, (byte)(asm.getCurrentOffset()-cloc+4));
        for (int i=0; i<count; ++i) {
            int target = targets[i];
            Integer t = new Integer(target);
            if (target <= i_start) {
                int offset = asm.getBranchTarget(t) - asm.getCurrentOffset() + 4;
                asm.emitDATA(offset);
            } else {
                asm.emitDATA(0x77777777);
                asm.recordForwardBranch(4, t);
            }
        }
    }
    public void visitLOOKUPSWITCH(int default_target, int[] values, int[] targets) {
        super.visitLOOKUPSWITCH(default_target, values, targets);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": LOOKUPSWITCH "+default_target)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitShort_Reg(x86.POP_r, EAX);
        for (int i=0; i<values.length; ++i) {
            int match = values[i];
            asm.emitARITH_Reg_Imm(x86.CMP_r_i32, EAX, match);
            int target = targets[i];
            branchHelper(CMP_EQ, target);
        }
        branchHelper(CMP_UNCOND, default_target);
    }
    private void RETURN4helper() {
        if (TraceMethods) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString("Leaving: "+method)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        // epilogue
        asm.emitShort_Reg(x86.POP_r, EAX); // store return value
        asm.emit1(x86.LEAVE);              // esp<-ebp, pop ebp
        asm.emit1_Imm16(x86.RET_i, (char)(n_paramwords<<2));
    }
    private void RETURN8helper() {
        if (TraceMethods) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString("Leaving: "+method)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        // epilogue
        asm.emitShort_Reg(x86.POP_r, EAX); // return value lo
        asm.emitShort_Reg(x86.POP_r, EDX); // return value hi
        asm.emit1(x86.LEAVE);              // esp<-ebp, pop ebp
        asm.emit1_Imm16(x86.RET_i, (char)(n_paramwords<<2));
    }
    public void visitIRETURN() {
        super.visitIRETURN();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": IRETURN")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        RETURN4helper();
    }
    public void visitLRETURN() {
        super.visitLRETURN();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": LRETURN")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        RETURN8helper();
    }
    public void visitFRETURN() {
        super.visitFRETURN();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": FRETURN")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        RETURN4helper();
    }
    public void visitDRETURN() {
        super.visitDRETURN();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": DRETURN")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        RETURN8helper();
    }
    public void visitARETURN() {
        super.visitARETURN();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": ARETURN")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        RETURN4helper();
    }
    public void visitVRETURN() {
        super.visitVRETURN();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": VRETURN")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        if (TraceMethods) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString("Leaving: "+method)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emit1(x86.LEAVE);              // esp<-ebp, pop ebp
        asm.emit1_Imm16(x86.RET_i, (char)(n_paramwords<<2));
    }
    public void GETSTATIC4helper(jq_StaticField f) {
        if (f.needsDynamicLink(method)) {
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": GETSTATIC4 "+f+" (dynpatch)")));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            // generate a runtime call, which will be backpatched.
            asm.startDynamicPatch(10); // 6
            asm.emitPUSH_i(Unsafe.addressOf(f));
            asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(x86ReferenceLinker._getstatic4));
            asm.endDynamicPatch();
        } else {
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": GETSTATIC4 "+f)));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            asm.emit2_Mem(x86.PUSH_m, f.getAddress());
        }
    }
    static int patch_getstatic4(int retloc, jq_StaticField f) {
        Unsafe.poke2(retloc-10, (short)0xFFEB);
        Unsafe.poke4(retloc-8, 0x35FF9090);
        Unsafe.poke4(retloc-4, f.getAddress());
        Unsafe.poke2(retloc-10, (short)0x9090);
        return 6;
    }
    public void GETSTATIC8helper(jq_StaticField f) {
        if (f.needsDynamicLink(method)) {
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": GETSTATIC8 "+f+" (dynpatch)")));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            // generate a runtime call, which will be backpatched.
            asm.startDynamicPatch(12);
            asm.emitPUSH_i(Unsafe.addressOf(f));
            asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(x86ReferenceLinker._getstatic8));
            asm.endDynamicPatch();
        } else {
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": GETSTATIC8 "+f)));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            asm.emit2_Mem(x86.PUSH_m, f.getAddress()+4); // hi
            asm.emit2_Mem(x86.PUSH_m, f.getAddress()  ); // lo
        }
    }
    static int patch_getstatic8(int retloc, jq_StaticField f) {
        Unsafe.poke2(retloc-10, (short)0xFFEB);
        Unsafe.poke4(retloc-8, f.getAddress()+4);
        Unsafe.poke2(retloc-4, (short)0x35FF);
        Unsafe.poke4(retloc-2, f.getAddress());
        Unsafe.poke2(retloc-10, (short)0x35FF);
        return 10;
    }
    public void visitIGETSTATIC(jq_StaticField f) {
        super.visitIGETSTATIC(f);
        GETSTATIC4helper(f);
    }
    public void visitLGETSTATIC(jq_StaticField f) {
        super.visitLGETSTATIC(f);
        GETSTATIC8helper(f);
    }
    public void visitFGETSTATIC(jq_StaticField f) {
        super.visitFGETSTATIC(f);
        GETSTATIC4helper(f);
    }
    public void visitDGETSTATIC(jq_StaticField f) {
        super.visitDGETSTATIC(f);
        GETSTATIC8helper(f);
    }
    public void visitAGETSTATIC(jq_StaticField f) {
        super.visitAGETSTATIC(f);
        GETSTATIC4helper(f);
    }
    public void PUTSTATIC4helper(jq_StaticField f) {
        if (f.needsDynamicLink(method)) {
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": PUTSTATIC4 "+f+" (dynpatch)")));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            // generate a runtime call, which will be backpatched.
            asm.startDynamicPatch(10); // 6
            asm.emitPUSH_i(Unsafe.addressOf(f));
            asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(x86ReferenceLinker._putstatic4));
            asm.endDynamicPatch();
        } else {
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": PUTSTATIC4 "+f)));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            asm.emit2_Mem(x86.POP_m, f.getAddress());
        }
    }
    static int patch_putstatic4(int retloc, jq_StaticField f) {
        Unsafe.poke2(retloc-10, (short)0xFFEB);
        Unsafe.poke4(retloc-8, 0x058F9090);
        Unsafe.poke4(retloc-4, f.getAddress());
        Unsafe.poke2(retloc-10, (short)0x9090);
        return 6;
    }
    public void PUTSTATIC8helper(jq_StaticField f) {
        if (f.needsDynamicLink(method)) {
            // generate a runtime call, which will be backpatched.
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": PUTSTATIC8 "+f+" (dynpatch)")));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            asm.startDynamicPatch(12);
            asm.emitPUSH_i(Unsafe.addressOf(f));
            asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(x86ReferenceLinker._putstatic8));
            asm.endDynamicPatch();
        } else {
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": PUTSTATIC8 "+f)));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            asm.emit2_Mem(x86.POP_m, f.getAddress()  ); // lo
            asm.emit2_Mem(x86.POP_m, f.getAddress()+4); // hi
        }
    }
    static int patch_putstatic8(int retloc, jq_StaticField f) {
        Unsafe.poke2(retloc-10, (short)0xFFEB);
        Unsafe.poke4(retloc-8, f.getAddress()+4);
        Unsafe.poke2(retloc-4, (short)0x058F);
        Unsafe.poke4(retloc-2, f.getAddress());
        Unsafe.poke2(retloc-10, (short)0x058F);
        return 10;
    }
    public void visitIPUTSTATIC(jq_StaticField f) {
        super.visitIPUTSTATIC(f);
        PUTSTATIC4helper(f);
    }
    public void visitLPUTSTATIC(jq_StaticField f) {
        super.visitLPUTSTATIC(f);
        PUTSTATIC8helper(f);
    }
    public void visitFPUTSTATIC(jq_StaticField f) {
        super.visitFPUTSTATIC(f);
        PUTSTATIC4helper(f);
    }
    public void visitDPUTSTATIC(jq_StaticField f) {
        super.visitDPUTSTATIC(f);
        PUTSTATIC8helper(f);
    }
    public void visitAPUTSTATIC(jq_StaticField f) {
        super.visitAPUTSTATIC(f);
        PUTSTATIC4helper(f);
    }
    public void GETFIELD1helper(jq_InstanceField f) {
        if (!f.getDeclaringClass().isPrepared()) {
            // generate a runtime call, which will be backpatched.
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": GETFIELD1 "+f+" (dynpatch)")));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            asm.startDynamicPatch(10); // 9
            asm.emitPUSH_i(Unsafe.addressOf(f));
            asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(x86ReferenceLinker._getfield1));
            asm.endDynamicPatch();
        } else {
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": GETFIELD1 "+f)));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            asm.emitShort_Reg(x86.POP_r, EAX); // obj ref
            asm.emit3_Reg_Mem(x86.MOVSX_r_m8, EBX, f.getOffset(), EAX);
            asm.emitShort_Reg(x86.PUSH_r, EBX);
        }
    }
    static int patch_getfield1(int retloc, jq_InstanceField f) {
        Unsafe.poke2(retloc-10, (short)0xFFEB);
        Unsafe.poke4(retloc-8, 0x0098BE0F);
        Unsafe.poke4(retloc-5, f.getOffset());
        Unsafe.poke1(retloc-1, (byte)0x53);
        Unsafe.poke2(retloc-10, (short)0x5890);
        return 9;
    }
    public void GETFIELD4helper(jq_InstanceField f) {
        if (!f.getDeclaringClass().isPrepared()) {
            // generate a runtime call, which will be backpatched.
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": GETFIELD4 "+f+" (dynpatch)")));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            asm.startDynamicPatch(10); // 7
            asm.emitPUSH_i(Unsafe.addressOf(f));
            asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(x86ReferenceLinker._getfield4));
            asm.endDynamicPatch();
        } else {
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": GETFIELD4 "+f)));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            asm.emitShort_Reg(x86.POP_r, EAX); // obj ref
            asm.emit2_Mem(x86.PUSH_m, f.getOffset(), EAX);
        }
    }
    static int patch_getfield4(int retloc, jq_InstanceField f) {
        Unsafe.poke2(retloc-10, (short)0xFFEB);
        Unsafe.poke4(retloc-8, 0xB0FF5890);
        Unsafe.poke4(retloc-4, f.getOffset());
        Unsafe.poke2(retloc-10, (short)0x9090);
        return 7;
    }
    public void GETFIELD8helper(jq_InstanceField f) {
        if (!f.getDeclaringClass().isPrepared()) {
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": GETFIELD8 "+f+" (dynpatch)")));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            // generate a runtime call, which will be backpatched.
            asm.startDynamicPatch(13);
            asm.emitPUSH_i(Unsafe.addressOf(f));
            asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(x86ReferenceLinker._getfield8));
            asm.endDynamicPatch();
        } else {
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": GETFIELD8 "+f)));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            asm.emitShort_Reg(x86.POP_r, EAX); // obj ref
            asm.emit2_Mem(x86.PUSH_m, f.getOffset()+4, EAX); // hi
            asm.emit2_Mem(x86.PUSH_m, f.getOffset(), EAX);   // lo
        }
    }
    static int patch_getfield8(int retloc, jq_InstanceField f) {
        Unsafe.poke4(retloc-10, 0x00B0FFEB);
        Unsafe.poke4(retloc-7, f.getOffset()+4);
        Unsafe.poke2(retloc-3, (short)0xB0FF);
        Unsafe.poke4(retloc-1, f.getOffset());
        Unsafe.poke1(retloc-10, (byte)0x58);
        return 10;
    }
    public void visitIGETFIELD(jq_InstanceField f) {
        super.visitIGETFIELD(f);
        GETFIELD4helper(f);
    }
    public void visitLGETFIELD(jq_InstanceField f) {
        super.visitLGETFIELD(f);
        GETFIELD8helper(f);
    }
    public void visitFGETFIELD(jq_InstanceField f) {
        super.visitFGETFIELD(f);
        GETFIELD4helper(f);
    }
    public void visitDGETFIELD(jq_InstanceField f) {
        super.visitDGETFIELD(f);
        GETFIELD8helper(f);
    }
    public void visitAGETFIELD(jq_InstanceField f) {
        super.visitAGETFIELD(f);
        GETFIELD4helper(f);
    }
    public void visitBGETFIELD(jq_InstanceField f) {
        super.visitBGETFIELD(f);
        GETFIELD1helper(f);
    }
    public void visitCGETFIELD(jq_InstanceField f) {
        super.visitCGETFIELD(f);
        if (!f.getDeclaringClass().isPrepared()) {
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": CGETFIELD "+f+" (dynpatch)")));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            // generate a runtime call, which will be backpatched.
            asm.startDynamicPatch(10); // 9
            asm.emitPUSH_i(Unsafe.addressOf(f));
            asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(x86ReferenceLinker._cgetfield));
            asm.endDynamicPatch();
        } else {
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": CGETFIELD "+f)));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            asm.emitShort_Reg(x86.POP_r, EAX); // obj ref
            asm.emit3_Reg_Mem(x86.MOVZX_r_m16, EBX, f.getOffset(), EAX);
            asm.emitShort_Reg(x86.PUSH_r, EBX);
        }
    }
    static int patch_cgetfield(int retloc, jq_InstanceField f) {
        Unsafe.poke2(retloc-10, (short)0xFFEB);
        Unsafe.poke4(retloc-8, 0x0098B70F);
        Unsafe.poke4(retloc-5, f.getOffset());
        Unsafe.poke1(retloc-1, (byte)0x53);
        Unsafe.poke2(retloc-10, (short)0x5890);
        return 9;
    }
    public void visitSGETFIELD(jq_InstanceField f) {
        super.visitSGETFIELD(f);
        if (!f.getDeclaringClass().isPrepared()) {
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": SGETFIELD "+f+" (dynpatch)")));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            // generate a runtime call, which will be backpatched.
            asm.startDynamicPatch(10); // 9
            asm.emitPUSH_i(Unsafe.addressOf(f));
            asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(x86ReferenceLinker._sgetfield));
            asm.endDynamicPatch();
        } else {
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": SGETFIELD "+f)));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            asm.emitShort_Reg(x86.POP_r, EAX); // obj ref
            asm.emit3_Reg_Mem(x86.MOVSX_r_m16, EBX, f.getOffset(), EAX);
            asm.emitShort_Reg(x86.PUSH_r, EBX);
        }
    }
    static int patch_sgetfield(int retloc, jq_InstanceField f) {
        Unsafe.poke2(retloc-10, (short)0xFFEB);
        Unsafe.poke4(retloc-8, 0x0098BF0F);
        Unsafe.poke4(retloc-5, f.getOffset());
        Unsafe.poke1(retloc-1, (byte)0x53);
        Unsafe.poke2(retloc-10, (short)0x5890);
        return 9;
    }
    public void visitZGETFIELD(jq_InstanceField f) {
        super.visitZGETFIELD(f);
        GETFIELD1helper(f);
    }
    public void PUTFIELD1helper(jq_InstanceField f) {
        if (!f.getDeclaringClass().isPrepared()) {
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": PUTFIELD1 "+f+" (dynpatch)")));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            // generate a runtime call, which will be backpatched.
            asm.startDynamicPatch(10); // 8
            asm.emitPUSH_i(Unsafe.addressOf(f));
            asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(x86ReferenceLinker._putfield1));
            asm.endDynamicPatch();
        } else {
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": PUTFIELD1 "+f)));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            // field has already been resolved.
            asm.emitShort_Reg(x86.POP_r, EBX);
            asm.emitShort_Reg(x86.POP_r, EAX);
            asm.emit2_Reg_Mem(x86.MOV_m_r8, EBX, f.getOffset(), EAX);
        }
    }
    static int patch_putfield1(int retloc, jq_InstanceField f) {
        Unsafe.poke2(retloc-10, (short)0xFFEB);
        Unsafe.poke4(retloc-8, 0x9888585B);
        Unsafe.poke4(retloc-4, f.getOffset());
        Unsafe.poke2(retloc-10, (short)0x9090);
        return 8;
    }
    public void PUTFIELD2helper(jq_InstanceField f) {
        if (!f.getDeclaringClass().isPrepared()) {
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": PUTFIELD2 "+f+" (dynpatch)")));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            // generate a runtime call, which will be backpatched.
            asm.startDynamicPatch(10); // 9
            asm.emitPUSH_i(Unsafe.addressOf(f));
            asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(x86ReferenceLinker._putfield2));
            asm.endDynamicPatch();
        } else {
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": PUTFIELD2 "+f)));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            // field has already been resolved.
            asm.emitShort_Reg(x86.POP_r, EBX);
            asm.emitShort_Reg(x86.POP_r, EAX);
            asm.emitprefix(x86.PREFIX_16BIT);
            asm.emit2_Reg_Mem(x86.MOV_m_r32, EBX, f.getOffset(), EAX);
        }
    }
    static int patch_putfield2(int retloc, jq_InstanceField f) {
        Unsafe.poke4(retloc-10, 0x6658FFEB);
        Unsafe.poke2(retloc-6, (short)0x9889);
        Unsafe.poke4(retloc-4, f.getOffset());
        Unsafe.poke2(retloc-10, (short)0x5B90);
        return 9;
    }
    public void PUTFIELD4helper(jq_InstanceField f) {
        if (!f.getDeclaringClass().isPrepared()) {
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": PUTFIELD4 "+f+" (dynpatch)")));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            // generate a runtime call, which will be backpatched.
            asm.startDynamicPatch(10); // 8
            asm.emitPUSH_i(Unsafe.addressOf(f));
            asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(x86ReferenceLinker._putfield4));
            asm.endDynamicPatch();
        } else {
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": PUTFIELD4 "+f)));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            // field has already been resolved.
            asm.emitShort_Reg(x86.POP_r, EBX);
            asm.emitShort_Reg(x86.POP_r, EAX);
            asm.emit2_Reg_Mem(x86.MOV_m_r32, EBX, f.getOffset(), EAX);
        }
    }
    static int patch_putfield4(int retloc, jq_InstanceField f) {
        Unsafe.poke2(retloc-10, (short)0xFFEB);
        Unsafe.poke4(retloc-8, 0x9889585B);
        Unsafe.poke4(retloc-4, f.getOffset());
        Unsafe.poke2(retloc-10, (short)0x9090);
        return 8;
    }
    public void PUTFIELD8helper(jq_InstanceField f) {
        if (!f.getDeclaringClass().isPrepared()) {
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": PUTFIELD8 "+f+" (dynpatch)")));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            // generate a runtime call, which will be backpatched.
            asm.startDynamicPatch(15);
            asm.emitPUSH_i(Unsafe.addressOf(f));
            asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(x86ReferenceLinker._putfield8));
            asm.endDynamicPatch();
        } else {
            if (TraceBytecodes) {
                asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": PUTFIELD8 "+f)));
                asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
            }
            // field has already been resolved.
            asm.emitShort_Reg(x86.POP_r, EBX); // lo
            asm.emitShort_Reg(x86.POP_r, EAX); // hi
            asm.emitShort_Reg(x86.POP_r, EDX);
            asm.emit2_Reg_Mem(x86.MOV_m_r32, EBX, f.getOffset()  , EDX); // lo
            asm.emit2_Reg_Mem(x86.MOV_m_r32, EAX, f.getOffset()+4, EDX); // hi
        }
    }
    static int patch_putfield8(int retloc, jq_InstanceField f) {
        Unsafe.poke4(retloc-10, 0x895AFFEB);
        Unsafe.poke1(retloc-6, (byte)0x9A);
        Unsafe.poke4(retloc-5, f.getOffset());
        Unsafe.poke2(retloc-1, (short)0x8289);
        Unsafe.poke4(retloc+1, f.getOffset()+4);
        Unsafe.poke2(retloc-10, (short)0x585B);
        return 10;
    }
    public void visitIPUTFIELD(jq_InstanceField f) {
        super.visitIPUTFIELD(f);
        PUTFIELD4helper(f);
    }
    public void visitLPUTFIELD(jq_InstanceField f) {
        super.visitLPUTFIELD(f);
        PUTFIELD8helper(f);
    }
    public void visitFPUTFIELD(jq_InstanceField f) {
        super.visitFPUTFIELD(f);
        PUTFIELD4helper(f);
    }
    public void visitDPUTFIELD(jq_InstanceField f) {
        super.visitDPUTFIELD(f);
        PUTFIELD8helper(f);
    }
    public void visitAPUTFIELD(jq_InstanceField f) {
        super.visitAPUTFIELD(f);
        PUTFIELD4helper(f);
    }
    public void visitBPUTFIELD(jq_InstanceField f) {
        super.visitBPUTFIELD(f);
        PUTFIELD1helper(f);
    }
    public void visitCPUTFIELD(jq_InstanceField f) {
        super.visitCPUTFIELD(f);
        PUTFIELD2helper(f);
    }
    public void visitSPUTFIELD(jq_InstanceField f) {
        super.visitSPUTFIELD(f);
        PUTFIELD2helper(f);
    }
    public void visitZPUTFIELD(jq_InstanceField f) {
        super.visitZPUTFIELD(f);
        PUTFIELD1helper(f);
    }
    private void INVOKEDPATCHhelper(byte op, jq_Method f) {
        int dpatchsize;
        jq_StaticMethod dpatchentry;
        switch (op) {
            case INVOKE_VIRTUAL:
                dpatchsize = 16;
                dpatchentry = x86ReferenceLinker._invokevirtual;
                break;
            case INVOKE_STATIC:
                dpatchsize = 11; // 5
                dpatchentry = x86ReferenceLinker._invokestatic;
                break;
            case INVOKE_SPECIAL:
                dpatchsize = 11; // 5
                dpatchentry = x86ReferenceLinker._invokespecial;
                break;
            case INVOKE_INTERFACE:
                // fallthrough
            default:
                throw new InternalError();
        }
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": INVOKE "+op+" "+f+" (dynpatch)")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        // generate a runtime call, which will be backpatched.
        asm.startDynamicPatch(dpatchsize);
        asm.emitPUSH_i(Unsafe.addressOf(f));
        asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(dpatchentry));
        asm.endDynamicPatch();
    }
    static int patch_invokevirtual(int retloc, jq_InstanceMethod f) {
        Unsafe.poke2(retloc-10, (short)0xFFEB);
        Unsafe.poke1(retloc-8, (byte)0x24);
        int objptroffset = (f.getParamWords() << 2) - 4;
        Unsafe.poke4(retloc-7, objptroffset);
        Unsafe.poke2(retloc-3, (short)0x588B);
        Unsafe.poke1(retloc-1, (byte)VTABLE_OFFSET);
        Unsafe.poke2(retloc, (short)0x93FF);
        Unsafe.poke4(retloc+2, f.getOffset());
        Unsafe.poke2(retloc-10, (short)0x848B);
        return 10;
    }
    static int patch_invokestatic(int retloc, jq_Method f) {
        Unsafe.poke4(retloc-10, 0x9090FFEB);
        Unsafe.poke2(retloc-6, (short)0xE890);
        Unsafe.poke4(retloc-4, f.getDefaultCompiledVersion().getEntrypoint()-retloc);
        Unsafe.poke2(retloc-10, (short)0x9090);
        return 5;
    }
    private void INVOKENODPATCHhelper(byte op, jq_Method f) {
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": INVOKE "+op+" "+f)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        switch(op) {
            case INVOKE_VIRTUAL: {
                int objptroffset = (f.getParamWords() << 2) - 4;
                int m_off = ((jq_InstanceMethod)f).getOffset();
                asm.emit2_Reg_Mem(x86.MOV_r_m32, EAX, objptroffset, ESP); // 7
                asm.emit2_Reg_Mem(x86.MOV_r_m32, EBX, VTABLE_OFFSET, EAX); // 3
                asm.emit2_Mem(x86.CALL_m, m_off, EBX); // 6
                break;
            }
            case INVOKE_SPECIAL:
                f = jq_Class.getInvokespecialTarget(clazz, (jq_InstanceMethod)f);
                // fallthrough
            case INVOKE_STATIC:
                asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(f));
                break;
            case INVOKE_INTERFACE:
                //jq.assert(jq.Bootstrapping || f.getDeclaringClass().isInterface());
                asm.emitPUSH_i(Unsafe.addressOf(f));
                asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(x86ReferenceLinker._invokeinterface));
                // need to pop args ourselves.
                asm.emit2_Reg_Mem(x86.LEA, ESP, f.getParamWords()<<2, ESP);
                break;
            default:
                jq.UNREACHABLE();
        }
    }
    private void INVOKEhelper(byte op, jq_Method f) {
        switch (op) {
            case INVOKE_VIRTUAL:
                if (!f.getDeclaringClass().isPrepared())
                    INVOKEDPATCHhelper(op, f);
                else
                    INVOKENODPATCHhelper(op, f);
                break;
            case INVOKE_STATIC:
                // fallthrough
            case INVOKE_SPECIAL:
                if (f.needsDynamicLink(method))
                    INVOKEDPATCHhelper(op, f);
                else
                    INVOKENODPATCHhelper(op, f);
                break;
            case INVOKE_INTERFACE:
                INVOKENODPATCHhelper(op, f);
                break;
            default:
                throw new InternalError();
        }
    }
    public void visitIINVOKE(byte op, jq_Method f) {
        super.visitIINVOKE(op, f);
        if (f.getDeclaringClass() == Unsafe._class) {
            gen_unsafe(f);
            return;
        }
        INVOKEhelper(op, f);
        asm.emitShort_Reg(x86.PUSH_r, EAX);
    }
    public void visitLINVOKE(byte op, jq_Method f) {
        super.visitLINVOKE(op, f);
        if (f.getDeclaringClass() == Unsafe._class) {
            gen_unsafe(f);
            return;
        }
        INVOKEhelper(op, f);
        asm.emitShort_Reg(x86.PUSH_r, EDX); // hi
        asm.emitShort_Reg(x86.PUSH_r, EAX); // lo
    }
    public void visitFINVOKE(byte op, jq_Method f) {
        super.visitFINVOKE(op, f);
        if (f.getDeclaringClass() == Unsafe._class) {
            gen_unsafe(f);
            return;
        }
        INVOKEhelper(op, f);
        asm.emitShort_Reg(x86.PUSH_r, EAX);
    }
    public void visitDINVOKE(byte op, jq_Method f) {
        super.visitDINVOKE(op, f);
        if (f.getDeclaringClass() == Unsafe._class) {
            gen_unsafe(f);
            return;
        }
        INVOKEhelper(op, f);
        asm.emitShort_Reg(x86.PUSH_r, EDX); // hi
        asm.emitShort_Reg(x86.PUSH_r, EAX); // lo
    }
    public void visitAINVOKE(byte op, jq_Method f) {
        super.visitAINVOKE(op, f);
        if (f.getDeclaringClass() == Unsafe._class) {
            gen_unsafe(f);
            return;
        }
        INVOKEhelper(op, f);
        asm.emitShort_Reg(x86.PUSH_r, EAX);
    }
    public void visitVINVOKE(byte op, jq_Method f) {
        super.visitVINVOKE(op, f);
        if (f.getDeclaringClass() == Unsafe._class) {
            gen_unsafe(f);
            return;
        }
        INVOKEhelper(op, f);
    }
    public void visitNEW(jq_Type f) {
        super.visitNEW(f);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": NEW "+f)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        if (f.isClassType() && !f.needsDynamicLink(method)) {
            jq_Class k = (jq_Class)f;
            asm.emitPUSH_i(k.getInstanceSize());
            asm.emitPUSH_i(Unsafe.addressOf(k.getVTable()));
            asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(SimpleAllocator._allocateObject));
        } else {
            asm.emitPUSH_i(Unsafe.addressOf(f));
            asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(Allocator._clsinitAndAllocateObject));
        }
        asm.emitShort_Reg(x86.PUSH_r, EAX);
    }
    public void visitNEWARRAY(jq_Array f) {
        super.visitNEWARRAY(f);
        if (!jq.Bootstrapping) {
            // initialize type now, to avoid backpatch.
            f.load(); f.verify(); f.prepare(); f.sf_initialize(); f.cls_initialize();
        }
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": NEWARRAY "+f)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        byte width = f.getLogElementSize();
        asm.emit2_Mem(x86.PUSH_m, 0, ESP);
        if (width != 0) asm.emit2_SHIFT_Mem_Imm8(x86.SHL_m32_i, 0, ESP, width);
        asm.emitARITH_Mem_Imm(x86.ADD_m_i32, 0, ESP, ObjectLayout.ARRAY_HEADER_SIZE);
        asm.emitPUSH_i(Unsafe.addressOf(f.getVTable()));
        asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(SimpleAllocator._allocateArray));
        asm.emitShort_Reg(x86.PUSH_r, EAX);
    }
    public void visitARRAYLENGTH() {
        super.visitARRAYLENGTH();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": ARRAYLENGTH")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitShort_Reg(x86.POP_r, EAX);
        asm.emit2_Mem(x86.PUSH_m, ARRAY_LENGTH_OFFSET, EAX);
    }
    public void visitATHROW() {
        super.visitATHROW();
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": ATHROW")));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(ExceptionDeliverer._athrow));
    }
    public void visitCHECKCAST(jq_Type f) {
        super.visitCHECKCAST(f);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": CHECKCAST "+f)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitPUSH_i(Unsafe.addressOf(f));
        asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(TypeCheck._checkcast));
        asm.emitShort_Reg(x86.PUSH_r, EAX);
    }
    public void visitINSTANCEOF(jq_Type f) {
        super.visitINSTANCEOF(f);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": INSTANCEOF "+f)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitPUSH_i(Unsafe.addressOf(f));
        asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(TypeCheck._instance_of));
        asm.emitShort_Reg(x86.PUSH_r, EAX);
    }
    public void visitMONITOR(byte op) {
        super.visitMONITOR(op);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": MONITOR "+op)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        int entry = (op == MONITOR_ENTER)?getMemberAddress(Monitor._monitorenter)
                                         :getMemberAddress(Monitor._monitorexit);
        asm.emitCALL_abs(x86.CALL_rel32, entry);
    }
    public void visitMULTINEWARRAY(jq_Type f, char dim) {
        super.visitMULTINEWARRAY(f, dim);
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": MULTINEWARRAY "+f+" "+dim)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        asm.emitPUSH_i(dim);
        asm.emitPUSH_i(Unsafe.addressOf(f));
        asm.emitCALL_abs(x86.CALL_rel32, getMemberAddress(Allocator._multinewarray));
        // pop dim args, because the callee doesn't do it.
        asm.emit2_Reg_Mem(x86.LEA, ESP, dim<<2, ESP);
        asm.emitShort_Reg(x86.PUSH_r, EAX);
    }

    private static Collection call_patches;
    static class CallPatch {
        jq_Method caller;
        int offset;
        jq_Member callee;
        CallPatch(jq_Method caller, int offset, jq_Member callee) {
            this.caller = caller;
            this.offset = offset;
            this.callee = callee;
        }
        public String toString() {
            return caller+" offset "+jq.hex(offset)+"->"+callee;
        }
    }
    public static void initCallPatches() {
        jq.assert(call_patches == null);
        call_patches = new LinkedList();
    }
    private int getMemberAddress(jq_Member m) {
        return getMemberAddress(m, method, asm);
    }
    private static int getMemberAddress(jq_Member m, jq_Method method, x86Assembler asm) {
        if (m instanceof jq_StaticField) {
            jq.assert(m.getDeclaringClass().isSFInitialized());
            jq_StaticField sf = (jq_StaticField)m;
            return sf.getAddress();
        }
        call_patches.add(new CallPatch(method, asm.getCurrentOffset()+1, m));
        return 0x66666666;
    }
    public static void patchCalls() {
        Iterator i = call_patches.iterator();
        while (i.hasNext()) {
            CallPatch cp = (CallPatch)i.next();
            jq.assert(cp.caller.getDefaultCompiledVersion() != null);
            int call_location = cp.caller.getDefaultCompiledVersion().getEntrypoint() + cp.offset;
            int callee_entry;
            if (cp.callee instanceof jq_Method) {
                jq_CompiledCode cc = ((jq_Method)cp.callee).getDefaultCompiledVersion();
                jq.assert(cc != null, cp.toString());
                callee_entry = ((jq_Method)cp.callee).getDefaultCompiledVersion().getEntrypoint();
            } else {
                jq.UNREACHABLE(); // handled in getMemberAddress. but should work anyway.
                callee_entry = ((jq_StaticField)cp.callee).getAddress();
            }
            //jq.assert(Unsafe.peek(call_location) == 0x66666666);
            Unsafe.poke4(call_location, callee_entry - call_location - 4);
        }
        call_patches = null;
    }
    
    private void gen_unsafe(jq_Method f) {
        if (TraceBytecodes) {
            asm.emitPUSH_i(Unsafe.addressOf(SystemInterface.toCString(i_start+": UNSAFE "+f)));
            asm.emit2_Mem(x86.CALL_m, SystemInterface._debugmsg.getAddress());
        }
        if ((f == Unsafe._addressOf) || (f == Unsafe._asObject) ||
            (f == Unsafe._floatToIntBits) || (f == Unsafe._intBitsToFloat) ||
            (f == Unsafe._doubleToLongBits) || (f == Unsafe._longBitsToDouble)) {
            asm.emit1(x86.NOP);
        } else if (f == Unsafe._peek) {
            asm.emitShort_Reg(x86.POP_r, EAX); // address
            asm.emit2_Mem(x86.PUSH_m, 0, EAX);
        } else if (f == Unsafe._poke1) {
            asm.emitShort_Reg(x86.POP_r, EBX); // value
            asm.emitShort_Reg(x86.POP_r, EAX); // address
            asm.emit2_Reg_Mem(x86.MOV_m_r8, EBX, 0, EAX);
        } else if (f == Unsafe._poke2) {
            asm.emitShort_Reg(x86.POP_r, EBX); // value
            asm.emitShort_Reg(x86.POP_r, EAX); // address
            asm.emitprefix(x86.PREFIX_16BIT);
            asm.emit2_Reg_Mem(x86.MOV_m_r32, EBX, 0, EAX);
        } else if (f == Unsafe._poke4) {
            asm.emitShort_Reg(x86.POP_r, EBX); // value
            asm.emitShort_Reg(x86.POP_r, EAX); // address
            asm.emit2_Reg_Mem(x86.MOV_m_r32, EBX, 0, EAX);
        } else if (f == Unsafe._getTypeOf) {
            asm.emitShort_Reg(x86.POP_r, EAX);
            asm.emit2_Reg_Mem(x86.MOV_r_m32, EBX, VTABLE_OFFSET, EAX);
            asm.emit2_Mem(x86.PUSH_m, 0, EBX);
        } else if (f == Unsafe._EAX) {
            asm.emitShort_Reg(x86.PUSH_r, EAX);
        } else if (f == Unsafe._EBP) {
            asm.emitShort_Reg(x86.PUSH_r, EBP);
        } else if (f == Unsafe._alloca) {
            asm.emitShort_Reg(x86.POP_r, EAX);
            asm.emitARITH_Reg_Reg(x86.SUB_r_r32, ESP, EAX);
        } else if (f == Unsafe._pushArg) {
            asm.emit1(x86.NOP);
        } else if (f == Unsafe._invoke) {
            asm.emitShort_Reg(x86.POP_r, EAX);
            asm.emit2_Reg(x86.CALL_r, EAX);
            asm.emitShort_Reg(x86.PUSH_r, EDX); // hi
            asm.emitShort_Reg(x86.PUSH_r, EAX); // lo
        } else if (f == Unsafe._getThreadBlock) {
            asm.emitprefix(x86.PREFIX_FS);
            asm.emit2_Mem(x86.PUSH_m, 0x14);
        } else if (f == Unsafe._setThreadBlock) {
            asm.emitprefix(x86.PREFIX_FS);
            asm.emit2_Mem(x86.POP_m, 0x14);
        } else if (f == Unsafe._switchRegisterState) {
            asm.emitShort_Reg(x86.POP_r, EAX); // sp
            asm.emitShort_Reg(x86.POP_r, EBP); // fp
            asm.emitShort_Reg(x86.POP_r, ECX); // ip
            asm.emit2_Reg_Reg(x86.MOV_r_r32, ESP, EAX);
            asm.emit2_Reg(x86.JMP_r, ECX);
        } else if (f == Unsafe._cas4) {
            asm.emitShort_Reg(x86.POP_r, EBX); // after
            asm.emitShort_Reg(x86.POP_r, EAX); // before
            asm.emitShort_Reg(x86.POP_r, ECX); // address
            asm.emitShort_Reg_Imm(x86.MOV_r_i32, EDX, 0);
            if (jq.SMP) asm.emitprefix(x86.PREFIX_LOCK);
            asm.emit3_Reg_Mem(x86.CMPXCHG_32, EBX, 0, ECX);
            asm.emitCJUMP_Short(x86.JNE, (byte)1);
            asm.emitShort_Reg(x86.INC_r32, EDX);
            asm.emitShort_Reg(x86.PUSH_r, EDX);
        } else {
            System.err.println(f.toString());
            jq.UNREACHABLE();
        }
    }

    public static final jq_StaticField _call_patches;
    static {
        jq_Class k = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LCompil3r/Reference/x86/x86ReferenceCompiler;");
        _call_patches = k.getOrCreateStaticField("call_patches", "Ljava/util/Collection;");
    }
}
