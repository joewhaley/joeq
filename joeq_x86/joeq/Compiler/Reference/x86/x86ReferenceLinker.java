/*
 * x86ReferenceLinker.java
 *
 * Created on January 1, 2001, 11:37 AM
 *
 * @author  jwhaley
 * @version 
 */

package Compil3r.Reference.x86;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Type;
import Clazz.jq_Reference;
import Clazz.jq_Array;
import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Clazz.jq_StaticField;
import Clazz.jq_CompiledCode;
import Clazz.jq_Method;
import Clazz.jq_InstanceMethod;
import Clazz.jq_StaticMethod;
import Clazz.jq_NameAndDesc;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import UTF.Utf8;
import jq;

public abstract class x86ReferenceLinker {

    public static /*final*/ boolean TRACE = false;
    
    static void patchCaller(jq_Method m, int retloc) {
        if ((Unsafe.peek(retloc-6)&0xFFFF) == 0xE890) {
            // patch static call
            Unsafe.poke4(retloc-4, m.getDefaultCompiledVersion().getEntrypoint()-retloc);
        }
        if (!m.isStatic() && ((jq_InstanceMethod)m).isVirtual()) {
            ((int[])m.getDeclaringClass().getVTable())[((jq_InstanceMethod)m).getOffset()>>2] = m.getDefaultCompiledVersion().getEntrypoint();
        }
    }
    
    static void getstatic4(jq_StaticField f) {
        f = f.resolve1();
        jq_Class k = f.getDeclaringClass();
        k.load(); k.verify(); k.prepare(); k.sf_initialize(); k.cls_initialize();
        int retloc = Unsafe.peek(Unsafe.EBP()+4);
        if (TRACE) SystemInterface.debugmsg("backpatching getstatic4 "+f+" ip: "+jq.hex8(retloc));
        int patchsize = x86ReferenceCompiler.patch_getstatic4(retloc, f);
        // change our return address to reexecute patched region
        Unsafe.poke4(Unsafe.EBP()+4, retloc-patchsize);
    }
    static void getstatic8(jq_StaticField f) {
        f = f.resolve1();
        jq_Class k = f.getDeclaringClass();
        k.load(); k.verify(); k.prepare(); k.sf_initialize(); k.cls_initialize();
        int retloc = Unsafe.peek(Unsafe.EBP()+4);
        if (TRACE) SystemInterface.debugmsg("backpatching getstatic8 "+f+" ip: "+jq.hex8(retloc));
        int patchsize = x86ReferenceCompiler.patch_getstatic8(retloc, f);
        // change our return address to reexecute patched region
        Unsafe.poke4(Unsafe.EBP()+4, retloc-patchsize);
    }
    static void putstatic4(jq_StaticField f) {
        f = f.resolve1();
        jq_Class k = f.getDeclaringClass();
        k.load(); k.verify(); k.prepare(); k.sf_initialize(); k.cls_initialize();
        int retloc = Unsafe.peek(Unsafe.EBP()+4);
        if (TRACE) SystemInterface.debugmsg("backpatching putstatic4 "+f+" ip: "+jq.hex8(retloc));
        int patchsize = x86ReferenceCompiler.patch_putstatic4(retloc, f);
        // change our return address to reexecute patched region
        Unsafe.poke4(Unsafe.EBP()+4, retloc-patchsize);
    }
    static void putstatic8(jq_StaticField f) {
        f = f.resolve1();
        jq_Class k = f.getDeclaringClass();
        k.load(); k.verify(); k.prepare(); k.sf_initialize(); k.cls_initialize();
        int retloc = Unsafe.peek(Unsafe.EBP()+4);
        if (TRACE) SystemInterface.debugmsg("backpatching putstatic8 "+f+" ip: "+jq.hex8(retloc));
        int patchsize = x86ReferenceCompiler.patch_putstatic8(retloc, f);
        // change our return address to reexecute patched region
        Unsafe.poke4(Unsafe.EBP()+4, retloc-patchsize);
    }
    static void getfield1(jq_InstanceField f) {
        f = f.resolve1();
        jq_Class k = f.getDeclaringClass();
        jq.assert(k.isClsInitialized());
        int retloc = Unsafe.peek(Unsafe.EBP()+4);
        if (TRACE) SystemInterface.debugmsg("backpatching getfield1 "+f+" ip: "+jq.hex8(retloc));
        int patchsize = x86ReferenceCompiler.patch_getfield1(retloc, f);
        // change our return address to reexecute patched region
        Unsafe.poke4(Unsafe.EBP()+4, retloc-patchsize);
    }
    static void cgetfield(jq_InstanceField f) {
        f = f.resolve1();
        jq_Class k = f.getDeclaringClass();
        jq.assert(k.isClsInitialized());
        int retloc = Unsafe.peek(Unsafe.EBP()+4);
        if (TRACE) SystemInterface.debugmsg("backpatching cgetfield "+f+" ip: "+jq.hex8(retloc));
        int patchsize = x86ReferenceCompiler.patch_cgetfield(retloc, f);
        // change our return address to reexecute patched region
        Unsafe.poke4(Unsafe.EBP()+4, retloc-patchsize);
    }
    static void sgetfield(jq_InstanceField f) {
        f = f.resolve1();
        jq_Class k = f.getDeclaringClass();
        jq.assert(k.isClsInitialized());
        int retloc = Unsafe.peek(Unsafe.EBP()+4);
        if (TRACE) SystemInterface.debugmsg("backpatching sgetfield "+f+" ip: "+jq.hex8(retloc));
        int patchsize = x86ReferenceCompiler.patch_sgetfield(retloc, f);
        // change our return address to reexecute patched region
        Unsafe.poke4(Unsafe.EBP()+4, retloc-patchsize);
    }
    static void getfield4(jq_InstanceField f) {
        f = f.resolve1();
        jq_Class k = f.getDeclaringClass();
        jq.assert(k.isClsInitialized());
        int retloc = Unsafe.peek(Unsafe.EBP()+4);
        if (TRACE) SystemInterface.debugmsg("backpatching getfield4 "+f+" ip: "+jq.hex8(retloc));
        int patchsize = x86ReferenceCompiler.patch_getfield4(retloc, f);
        // change our return address to reexecute patched region
        Unsafe.poke4(Unsafe.EBP()+4, retloc-patchsize);
    }
    static void getfield8(jq_InstanceField f) {
        f = f.resolve1();
        jq_Class k = f.getDeclaringClass();
        jq.assert(k.isClsInitialized());
        int retloc = Unsafe.peek(Unsafe.EBP()+4);
        if (TRACE) SystemInterface.debugmsg("backpatching getfield8 "+f+" ip: "+jq.hex8(retloc));
        int patchsize = x86ReferenceCompiler.patch_getfield8(retloc, f);
        // change our return address to reexecute patched region
        Unsafe.poke4(Unsafe.EBP()+4, retloc-patchsize);
    }
    static void putfield1(jq_InstanceField f) {
        f = f.resolve1();
        jq_Class k = f.getDeclaringClass();
        jq.assert(k.isClsInitialized());
        int retloc = Unsafe.peek(Unsafe.EBP()+4);
        if (TRACE) SystemInterface.debugmsg("backpatching putfield1 "+f+" ip: "+jq.hex8(retloc));
        int patchsize = x86ReferenceCompiler.patch_putfield1(retloc, f);
        // change our return address to reexecute patched region
        Unsafe.poke4(Unsafe.EBP()+4, retloc-patchsize);
    }
    static void putfield2(jq_InstanceField f) {
        f = f.resolve1();
        jq_Class k = f.getDeclaringClass();
        jq.assert(k.isClsInitialized());
        int retloc = Unsafe.peek(Unsafe.EBP()+4);
        if (TRACE) SystemInterface.debugmsg("backpatching putfield2 "+f+" ip: "+jq.hex8(retloc));
        int patchsize = x86ReferenceCompiler.patch_putfield2(retloc, f);
        // change our return address to reexecute patched region
        Unsafe.poke4(Unsafe.EBP()+4, retloc-patchsize);
    }
    static void putfield4(jq_InstanceField f) {
        f = f.resolve1();
        jq_Class k = f.getDeclaringClass();
        jq.assert(k.isClsInitialized());
        int retloc = Unsafe.peek(Unsafe.EBP()+4);
        if (TRACE) SystemInterface.debugmsg("backpatching putfield4 "+f+" ip: "+jq.hex8(retloc));
        int patchsize = x86ReferenceCompiler.patch_putfield4(retloc, f);
        // change our return address to reexecute patched region
        Unsafe.poke4(Unsafe.EBP()+4, retloc-patchsize);
    }
    static void putfield8(jq_InstanceField f) {
        f = f.resolve1();
        jq_Class k = f.getDeclaringClass();
        jq.assert(k.isClsInitialized());
        int retloc = Unsafe.peek(Unsafe.EBP()+4);
        if (TRACE) SystemInterface.debugmsg("backpatching putfield8 "+f+" ip: "+jq.hex8(retloc));
        int patchsize = x86ReferenceCompiler.patch_putfield8(retloc, f);
        // change our return address to reexecute patched region
        Unsafe.poke4(Unsafe.EBP()+4, retloc-patchsize);
    }
    static void invokevirtual(jq_InstanceMethod f) {
        f = f.resolve1();
        jq_Class k = f.getDeclaringClass();
        jq.assert(k.isClsInitialized());
        int retloc = Unsafe.peek(Unsafe.EBP()+4);
        if (TRACE) SystemInterface.debugmsg("backpatching invokevirtual "+f+" ip: "+jq.hex8(retloc));
        int patchsize = x86ReferenceCompiler.patch_invokevirtual(retloc, f);
        // change our return address to reexecute patched region
        Unsafe.poke4(Unsafe.EBP()+4, retloc-patchsize);
    }
    static void invokestatic(jq_Method f) {
        f = (jq_Method)f.resolve();
        jq_Class k = f.getDeclaringClass();
        k.load(); k.verify(); k.prepare(); k.sf_initialize(); k.cls_initialize();
        int retloc = Unsafe.peek(Unsafe.EBP()+4);
        if (TRACE) SystemInterface.debugmsg("backpatching invokestatic "+f+" ip: "+jq.hex8(retloc));
        int patchsize = x86ReferenceCompiler.patch_invokestatic(retloc, f);
        // change our return address to reexecute patched region
        Unsafe.poke4(Unsafe.EBP()+4, retloc-patchsize);
    }
    static void invokespecial(jq_InstanceMethod f) {
        f = f.resolve1();
        jq_Class k = f.getDeclaringClass();
        k.load(); k.verify(); k.prepare(); k.sf_initialize(); k.cls_initialize();
        f = jq_Class.getInvokespecialTarget(k, f);
        int retloc = Unsafe.peek(Unsafe.EBP()+4);
        if (TRACE) SystemInterface.debugmsg("backpatching invokespecial "+f+" ip: "+jq.hex8(retloc));
        // special invocation is now directly bound.
        int patchsize = x86ReferenceCompiler.patch_invokestatic(retloc, f);
        // change our return address to reexecute patched region
        Unsafe.poke4(Unsafe.EBP()+4, retloc-patchsize);
    }
    static long invokeinterface(jq_InstanceMethod f) throws Throwable {
        f = f.resolve1();
        int n_paramwords = f.getParamWords();
        int obj_location = Unsafe.EBP() + ((n_paramwords+2)<<2);
        Object o = Unsafe.asObject(Unsafe.peek(obj_location));
        jq_Reference t = Unsafe.getTypeOf(o);
        if (!t.implementsInterface(f.getDeclaringClass()))
            throw new IncompatibleClassChangeError(t+" does not implement interface "+f.getDeclaringClass());
        jq_InstanceMethod m = t.getVirtualMethod(f.getNameAndDesc());
        if (m == null)
            throw new AbstractMethodError();
        //if (TRACE) SystemInterface.debugmsg("invokeinterface "+f+" on object type "+t+" resolved to "+m);
        jq_Class k = m.getDeclaringClass();
        k.sf_initialize(); k.cls_initialize();
        for (int i=0; i<n_paramwords; ++i) {
            Unsafe.pushArg(Unsafe.peek(Unsafe.EBP() + ((n_paramwords-i+2)<<2)));
        }
        return Unsafe.invoke(m.getDefaultCompiledVersion().getEntrypoint());
    }
    static void abstractMethodError() throws AbstractMethodError {
        SystemInterface.debugmsg("Unimplemented abstract method!");
        throw new AbstractMethodError();
    }
    static void nativeMethodError() throws LinkageError {
        SystemInterface.debugmsg("Unimplemented native method!");
        throw new LinkageError();
    }
    
    public static final jq_Class _class;
    public static final jq_StaticMethod _getstatic4;
    public static final jq_StaticMethod _getstatic8;
    public static final jq_StaticMethod _putstatic4;
    public static final jq_StaticMethod _putstatic8;
    public static final jq_StaticMethod _getfield1;
    public static final jq_StaticMethod _cgetfield;
    public static final jq_StaticMethod _sgetfield;
    public static final jq_StaticMethod _getfield4;
    public static final jq_StaticMethod _getfield8;
    public static final jq_StaticMethod _putfield1;
    public static final jq_StaticMethod _putfield2;
    public static final jq_StaticMethod _putfield4;
    public static final jq_StaticMethod _putfield8;
    public static final jq_StaticMethod _invokevirtual;
    public static final jq_StaticMethod _invokestatic;
    public static final jq_StaticMethod _invokespecial;
    public static final jq_StaticMethod _invokeinterface;
    public static final jq_StaticMethod _abstractMethodError;
    public static final jq_StaticMethod _nativeMethodError;
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LCompil3r/Reference/x86/x86ReferenceLinker;");
        _getstatic4 = _class.getOrCreateStaticMethod("getstatic4", "(LClazz/jq_StaticField;)V");
        _getstatic8 = _class.getOrCreateStaticMethod("getstatic8", "(LClazz/jq_StaticField;)V");
        _putstatic4 = _class.getOrCreateStaticMethod("putstatic4", "(LClazz/jq_StaticField;)V");
        _putstatic8 = _class.getOrCreateStaticMethod("putstatic8", "(LClazz/jq_StaticField;)V");
        _getfield1 = _class.getOrCreateStaticMethod("getfield1", "(LClazz/jq_InstanceField;)V");
        _sgetfield = _class.getOrCreateStaticMethod("sgetfield", "(LClazz/jq_InstanceField;)V");
        _cgetfield = _class.getOrCreateStaticMethod("cgetfield", "(LClazz/jq_InstanceField;)V");
        _getfield4 = _class.getOrCreateStaticMethod("getfield4", "(LClazz/jq_InstanceField;)V");
        _getfield8 = _class.getOrCreateStaticMethod("getfield8", "(LClazz/jq_InstanceField;)V");
        _putfield1 = _class.getOrCreateStaticMethod("putfield1", "(LClazz/jq_InstanceField;)V");
        _putfield2 = _class.getOrCreateStaticMethod("putfield2", "(LClazz/jq_InstanceField;)V");
        _putfield4 = _class.getOrCreateStaticMethod("putfield4", "(LClazz/jq_InstanceField;)V");
        _putfield8 = _class.getOrCreateStaticMethod("putfield8", "(LClazz/jq_InstanceField;)V");
        _invokevirtual = _class.getOrCreateStaticMethod("invokevirtual", "(LClazz/jq_InstanceMethod;)V");
        _invokestatic = _class.getOrCreateStaticMethod("invokestatic", "(LClazz/jq_Method;)V");
        _invokespecial = _class.getOrCreateStaticMethod("invokespecial", "(LClazz/jq_InstanceMethod;)V");
        _invokeinterface = _class.getOrCreateStaticMethod("invokeinterface", "(LClazz/jq_InstanceMethod;)J");
        _abstractMethodError = _class.getOrCreateStaticMethod("abstractMethodError", "()V");
        _nativeMethodError = _class.getOrCreateStaticMethod("nativeMethodError", "()V");
    }
}
