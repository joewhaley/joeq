// CompilationState.java, created Oct 4, 2003 11:09:20 PM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r;

import java.util.Collection;
import java.util.Set;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_Member;
import Clazz.jq_Method;
import Clazz.jq_Type;
import Main.jq;
import Run_Time.TypeCheck;
import UTF.Utf8;

/**
 * CompilationState
 * 
 * @author John Whaley
 * @version $Id$
 */
public abstract class CompilationState implements CompilationConstants {
    
    /** Default compilation state object.
     * This is here temporarily until we remove all static calls in the compiler. */
    public static CompilationState DEFAULT;
    
    public static final boolean VerifyAssertions = true;
    
    public abstract boolean needsDynamicLink(jq_Method method, jq_Member member);
    public abstract boolean needsDynamicLink(jq_Method method, jq_Type type);
    
    public abstract jq_Member tryResolve(jq_Member m);
    public abstract jq_Member resolve(jq_Member m);
    
    public abstract byte isSubtype(jq_Type t1, jq_Type t2);
    public abstract jq_Type findCommonSuperclass(jq_Type t1, jq_Type t2);
    
    public abstract byte declaresInterface(jq_Class klass, Collection interfaces);
    public abstract byte implementsInterface(jq_Class klass, jq_Class inter);
    
    public abstract jq_Type getOrCreateType(Utf8 desc);
    
    static {
        if (jq.nullVM) {
            DEFAULT = new StaticCompilation();
        } else if (jq.IsBootstrapping) {
            DEFAULT = new BootstrapCompilation();
        } else {
            DEFAULT = new DynamicCompilation();
        }
    }
    
    public static class StaticCompilation extends CompilationState {

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#needsDynamicLink(Clazz.jq_Method, Clazz.jq_Member)
         */
        public boolean needsDynamicLink(jq_Method method, jq_Member member) {
            if (member.isPrepared()) return false;
            member.getDeclaringClass().prepare();
            return !member.isPrepared();
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#needsDynamicLink(Clazz.jq_Method, Clazz.jq_Type)
         */
        public boolean needsDynamicLink(jq_Method method, jq_Type type) {
            type.prepare();
            return !type.isPrepared();
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#tryResolve(Clazz.jq_Member)
         */
        public jq_Member tryResolve(jq_Member m) {
            return m.resolve();
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#resolve(Clazz.jq_Member)
         */
        public jq_Member resolve(jq_Member m) {
            try {
                m = m.resolve();
            } catch (Error _) { }
            return m;
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#isSubtype(Clazz.jq_Type, Clazz.jq_Type)
         */
        public byte isSubtype(jq_Type t1, jq_Type t2) {
            return t1.isSubtypeOf(t2) ? YES : NO;
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#findCommonSuperclass(Clazz.jq_Type, Clazz.jq_Type)
         */
        public jq_Type findCommonSuperclass(jq_Type t1, jq_Type t2) {
            return TypeCheck.findCommonSuperclass(t1, t2, true);
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#declaresInterface(Clazz.jq_Class, java.util.Collection)
         */
        public byte declaresInterface(jq_Class klass, Collection interfaces) {
            return TypeCheck.declaresInterface(klass, interfaces, true);
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#implementsInterface(Clazz.jq_Class, Clazz.jq_Class)
         */
        public byte implementsInterface(jq_Class klass, jq_Class inter) {
            return klass.isSubtypeOf(inter) ? YES : NO;
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#getOrCreateType(UTF.Utf8)
         */
        public jq_Type getOrCreateType(Utf8 desc) {
            return PrimordialClassLoader.loader.getOrCreateBSType(desc);
        }
        
    }
    
    public static class BootstrapCompilation extends CompilationState {

        Set/*<jq_Type>*/ boot_types;

        public void setBootTypes(Set boot_types) {
            this.boot_types = boot_types;
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#needsDynamicLink(Clazz.jq_Method, Clazz.jq_Member)
         */
        public boolean needsDynamicLink(jq_Method method, jq_Member member) {
            if (member.isPrepared()) return false;
            return boot_types == null ||
                   !boot_types.contains(member.getDeclaringClass());
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#needsDynamicLink(Clazz.jq_Method, Clazz.jq_Type)
         */
        public boolean needsDynamicLink(jq_Method method, jq_Type type) {
            if (type.isPrepared()) return false;
            return boot_types == null ||
                   !boot_types.contains(type);
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#tryResolve(Clazz.jq_Member)
         */
        public jq_Member tryResolve(jq_Member m) {
            try {
                m = m.resolve();
            } catch (Error _) { }
            return m;
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#resolve(Clazz.jq_Member)
         */
        public jq_Member resolve(jq_Member m) {
            return m.resolve();
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#isSubtype(Clazz.jq_Type, Clazz.jq_Type)
         */
        public byte isSubtype(jq_Type t1, jq_Type t2) {
            return TypeCheck.isAssignable_noload(t1, t2);
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#findCommonSuperclass(Clazz.jq_Type, Clazz.jq_Type)
         */
        public jq_Type findCommonSuperclass(jq_Type t1, jq_Type t2) {
            return TypeCheck.findCommonSuperclass(t1, t2, false);
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#declaresInterface(Clazz.jq_Class, java.util.Collection)
         */
        public byte declaresInterface(jq_Class klass, Collection interfaces) {
            return TypeCheck.declaresInterface(klass, interfaces, false);
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#implementsInterface(Clazz.jq_Class, Clazz.jq_Class)
         */
        public byte implementsInterface(jq_Class klass, jq_Class inter) {
            return TypeCheck.implementsInterface_noload(klass, inter);
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#getOrCreateType(UTF.Utf8)
         */
        public jq_Type getOrCreateType(Utf8 desc) {
            return PrimordialClassLoader.loader.getOrCreateBSType(desc);
        }
        
    }
    
    public static class DynamicCompilation extends CompilationState {

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#needsDynamicLink(Clazz.jq_Method, Clazz.jq_Member)
         */
        public boolean needsDynamicLink(jq_Method method, jq_Member member) {
            if (member.isStatic() &&
                method.getDeclaringClass() != member.getDeclaringClass() &&
                !member.getDeclaringClass().isClsInitialized())
                return true;
            if (member instanceof jq_Method)
                return !member.isInitialized();
            return !member.isPrepared();
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#needsDynamicLink(Clazz.jq_Method, Clazz.jq_Type)
         */
        public boolean needsDynamicLink(jq_Method method, jq_Type type) {
            return method.getDeclaringClass() != type &&
                   !type.isClsInitialized();
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#tryResolve(Clazz.jq_Member)
         */
        public jq_Member tryResolve(jq_Member m) {
            if (m.getDeclaringClass().isPrepared()) {
                try {
                    m = m.resolve();
                } catch (Error _) { }
            }
            return m;
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#resolve(Clazz.jq_Member)
         */
        public jq_Member resolve(jq_Member m) {
            try {
                m = m.resolve();
            } catch (Error _) { }
            return m;
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#isSubtype(Clazz.jq_Type, Clazz.jq_Type)
         */
        public byte isSubtype(jq_Type t1, jq_Type t2) {
            return TypeCheck.isAssignable_noload(t1, t2);
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#findCommonSuperclass(Clazz.jq_Type, Clazz.jq_Type)
         */
        public jq_Type findCommonSuperclass(jq_Type t1, jq_Type t2) {
            return TypeCheck.findCommonSuperclass(t1, t2, false);
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#declaresInterface(Clazz.jq_Class, java.util.Collection)
         */
        public byte declaresInterface(jq_Class klass, Collection interfaces) {
            return TypeCheck.declaresInterface(klass, interfaces, false);
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#implementsInterface(Clazz.jq_Class, Clazz.jq_Class)
         */
        public byte implementsInterface(jq_Class klass, jq_Class inter) {
            return TypeCheck.implementsInterface_noload(klass, inter);
        }

        /* (non-Javadoc)
         * @see Compil3r.CompilationState#getOrCreateType(UTF.Utf8)
         */
        public jq_Type getOrCreateType(Utf8 desc) {
            return PrimordialClassLoader.loader.getOrCreateBSType(desc);
        }
    }
    
}
