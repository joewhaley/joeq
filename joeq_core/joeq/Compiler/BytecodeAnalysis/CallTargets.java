/*
 * CallTargets.java
 *
 * Created on June 27, 2001, 9:26 PM
 *
 * @author  John Whaley
 * @version 
 */

package Compil3r.BytecodeAnalysis;

import Bootstrap.PrimordialClassLoader;
import Clazz.*;
import Run_Time.TypeCheck;
import Util.LinearSet;
import Util.NullIterator;
import Util.SingletonIterator;
import jq;
import java.util.*;

public abstract class CallTargets {

    public static CallTargets getTargets(jq_Class callingClass,
                                         jq_Method method,
                                         byte type,
                                         Set possibleReceiverTypes,
                                         boolean loadClasses)
    {
        if (type == BytecodeVisitor.INVOKE_STATIC) return getStaticTargets((jq_StaticMethod)method);
        jq_InstanceMethod imethod = (jq_InstanceMethod)method;
        if (type == BytecodeVisitor.INVOKE_SPECIAL) return getSpecialTargets(callingClass, imethod, loadClasses);
        jq.assert(type == BytecodeVisitor.INVOKE_VIRTUAL || type == BytecodeVisitor.INVOKE_INTERFACE);

        Set c = new LinearSet();
        boolean complete = true;
        if ((type == BytecodeVisitor.INVOKE_VIRTUAL) && imethod.getDeclaringClass().isPrepared()) {
            // fast search using vtable
            jq.assert(!imethod.getDeclaringClass().isInterface());
            int offset = imethod.getOffset() >>> 2;
            Iterator i = possibleReceiverTypes.iterator();
            while (i.hasNext()) {
                jq_Reference rtype = (jq_Reference)i.next();
                if (rtype.isClassType()) {
                    jq_Class rclass = (jq_Class)rtype;
                    jq.assert(!rclass.isAbstract());
                    if (!rclass.isPrepared()) {
                        jq_InstanceMethod target;
                        for (;;) {
                            if (!rclass.isLoaded()) {
                                if (!loadClasses) {
                                    complete = false; // conservative.
                                    continue;
                                }
                                rclass.load();
                            }
                            target = (jq_InstanceMethod)rclass.getDeclaredMember(imethod.getNameAndDesc());
                            if (target != null) break;
                            rclass = rclass.getSuperclass();
                            jq.assert(rclass != null);
                        }
                        jq.assert(imethod.getNameAndDesc().equals(target.getNameAndDesc()));
                        jq.assert(!target.isAbstract());
                        c.add(target);
                        continue;
                    }
                    jq_InstanceMethod target = rclass.getVirtualMethods()[offset-1];
                    jq.assert(imethod.getNameAndDesc().equals(target.getNameAndDesc()), imethod+" != "+target);
                    jq.assert(!target.isAbstract());
                    c.add(target);
                } else {
                    jq.assert(rtype.isArrayType());
                    jq.assert(imethod.getDeclaringClass() == PrimordialClassLoader.loader.getJavaLangObject());
                    jq.assert(!imethod.isAbstract());
                    c.add(imethod);
                }
            }
        } else {
            // slow search.
            Iterator i = possibleReceiverTypes.iterator();
            while (i.hasNext()) {
                jq_Reference rtype = (jq_Reference)i.next();
                if (rtype.isClassType()) {
                    jq_Class rclass = (jq_Class)rtype;
                    jq_InstanceMethod target;
                    for (;;) {
                        if (!loadClasses && !rclass.isLoaded()) {
                            complete = false; // conservative.
                            continue; 
                        }
                        target = (jq_InstanceMethod)rclass.getDeclaredMember(imethod.getNameAndDesc());
                        if (target != null) break;
                        rclass = rclass.getSuperclass();
                        jq.assert(rclass != null);
                    }
                    jq.assert(imethod.getNameAndDesc().equals(target.getNameAndDesc()));
                    jq.assert(!target.isAbstract());
                    c.add(target);
                    continue;
                } else {
                    jq.assert(rtype.isArrayType());
                    jq.assert(imethod.getDeclaringClass() == PrimordialClassLoader.loader.getJavaLangObject());
                    jq.assert(!imethod.isAbstract());
                    c.add(imethod);
                }
            }
        }
        return new MultipleCallTargets(c, complete);
    }

    public static CallTargets getTargets(jq_Class callingClass,
                                         jq_Method method,
                                         byte type,
                                         jq_Reference receiverType,
                                         boolean exact,
                                         boolean loadClasses)
    {
        if (type == BytecodeVisitor.INVOKE_STATIC) return getStaticTargets((jq_StaticMethod)method);
        jq_InstanceMethod imethod = (jq_InstanceMethod)method;
        if (type == BytecodeVisitor.INVOKE_SPECIAL) return getSpecialTargets(callingClass, imethod, loadClasses);
        jq.assert(type == BytecodeVisitor.INVOKE_VIRTUAL || type == BytecodeVisitor.INVOKE_INTERFACE);

        if (receiverType.isArrayType()) {
            jq.assert(imethod.getDeclaringClass() == PrimordialClassLoader.loader.getJavaLangObject());
            return new SingleCallTarget(imethod, true);
        }
        
        jq_Class rclass = (jq_Class)receiverType;
        
        if (loadClasses) {
            if (TypeCheck.isSuperclassOf(rclass, imethod.getDeclaringClass())) {
                // errr... rclass is a supertype of the method's class!
                receiverType = rclass = imethod.getDeclaringClass();
            }
        }
        
        if (exact) {
            if (!loadClasses) {
                if (!rclass.isLoaded()) return NoCallTarget.INSTANCE;
                rclass.load();
            }
            for (;;) {
                jq_InstanceMethod target = (jq_InstanceMethod)rclass.getDeclaredMember(imethod.getNameAndDesc());
                if (target != null) return new SingleCallTarget(target, true);
                jq.assert(rclass != imethod.getDeclaringClass());
                rclass = rclass.getSuperclass();
                jq.assert(rclass != null, imethod+" not found in "+receiverType);
            }
        }

	// TEMPORARY HACK: sometimes casts from interface types are
	// lost in the type analysis, leading to virtual calls on interfaces
	if (loadClasses) {
	    rclass.load();
	    if (rclass.isInterface() && type == BytecodeVisitor.INVOKE_VIRTUAL) {
		jq.assert(!imethod.getDeclaringClass().isInterface());
		receiverType = rclass = imethod.getDeclaringClass();
	    }
	}

        Set c = new LinearSet();
        boolean complete = true;
        if (type == BytecodeVisitor.INVOKE_VIRTUAL) {
            if (imethod.getDeclaringClass().isPrepared()) {
                // fast search.
                jq.assert(!imethod.getDeclaringClass().isInterface());
                int offset = imethod.getOffset() >>> 2;
                if (!rclass.isPrepared()) {
                    for (;;) {
                        if (!rclass.isLoaded()) {
                            if (!loadClasses) {
                                complete = false; // conservative.
                                break;
                            }
                            rclass.load();
                        }
                        jq_InstanceMethod target = (jq_InstanceMethod)rclass.getDeclaredMember(imethod.getNameAndDesc());
                        if (target != null) {
                            if (!target.isAbstract()) c.add(target);
                            break;
                        }
                        jq.assert(rclass != imethod.getDeclaringClass());
                        rclass = rclass.getSuperclass();
                        jq.assert(rclass != null);
                    }
                }
                jq_InstanceMethod target;
                Stack subclass = new Stack();
                subclass.push(receiverType);
                while (!subclass.empty()) {
                    rclass = (jq_Class)subclass.pop();
                    if (!rclass.isLoaded()) {
                        if (!loadClasses) {
                            complete = false; // conservative.
                            continue;
                        }
                        rclass.load();
                    }
                    if (!rclass.isPrepared()) {
                        target = (jq_InstanceMethod)rclass.getDeclaredMember(imethod.getNameAndDesc());
                        if ((target != null) && !target.isAbstract()) c.add(target);
                    } else {
			jq.assert(offset-1 >= 0 && offset-1 < rclass.getVirtualMethods().length, "bad offset "+(offset-1)+" class "+rclass+" method "+imethod);
                        target = rclass.getVirtualMethods()[offset-1];
                        jq.assert(imethod.getNameAndDesc().equals(target.getNameAndDesc()), imethod+" != "+target);
                        if (!target.isAbstract()) c.add(target);
                    }
                    if (target != null) {
                        if (!target.isFinal() && !target.isPrivate()) {
                            if (!rclass.isFinal()) {
                                complete = false; // conservative.
                            }
                            jq_Class[] subclasses = rclass.getSubClasses();
                            for (int i=0; i<subclasses.length; ++i) subclass.push(subclasses[i]);
                        }
                    } else {
                        if (!rclass.isFinal()) {
                            complete = false; // conservative.
                        }
                    }
                }
                return new MultipleCallTargets(c, complete);
            }
        }
        
        if (type == BytecodeVisitor.INVOKE_INTERFACE) {
            if (loadClasses) {
                rclass.load();
            }
            if (!rclass.isLoaded() || ((jq_Class)rclass).isInterface()) {
                // not the true receiver type, or we don't know anything about the receiver type because
                // it isn't loaded, so we fall back to the case where we don't consider the receiver
                // (calls to java.lang.Object methods on interfaces should have been caught above
                //  because they are virtual calls, and java.lang.Object is always prepared)
                return getTargets(callingClass, method, type, loadClasses);
            }
        }
        
        // slow search, interface or virtual call.  instance method is not prepared.
        for (;;) {
            if (!loadClasses && !rclass.isLoaded()) {
                complete = false; // conservative.
                break;
            } else {
                rclass.load();
            }
            jq_InstanceMethod target = (jq_InstanceMethod)rclass.getDeclaredMember(imethod.getNameAndDesc());
            if (target != null) {
                if (!target.isAbstract()) c.add(target);
                break;
            }
            jq.assert(rclass != imethod.getDeclaringClass());
            rclass = rclass.getSuperclass();
            jq.assert(rclass != null);
        }
        Stack subclass = new Stack();
        subclass.push(receiverType);
        while (!subclass.empty()) {
            rclass = (jq_Class)subclass.pop();
            if (!rclass.isLoaded()) {
                if (!loadClasses) {
                    complete = false; // conservative.
                    continue;
                }
                rclass.load();
            }
            jq_InstanceMethod target = (jq_InstanceMethod)rclass.getDeclaredMember(imethod.getNameAndDesc());
            if (target != null) {
                if (!target.isAbstract()) c.add(target);
                if (!target.isFinal() && !target.isPrivate()) {
                    if (!rclass.isFinal()) {
                        complete = false; // conservative.
                    }
                    jq_Class[] subclasses = rclass.getSubClasses();
                    for (int i=0; i<subclasses.length; ++i) subclass.push(subclasses[i]);
                }
            } else {
                if (!rclass.isFinal()) {
                    complete = false; // conservative.
                }
            }
        }
        return new MultipleCallTargets(c, complete);
    }

    static final byte YES = 1;
    static final byte MAYBE = 2;
    static final byte NO = 3;
    
    static byte implementsInterface_noload(jq_Class klass, jq_Class inter) {
        byte res = NO; jq_Class k = klass;
        if (!klass.isLoaded()) return MAYBE;
        do {
            if (k.getDeclaredInterface(inter.getDesc()) == inter) return YES;
            k = k.getSuperclass();
            if (!k.isLoaded()) {
                res = MAYBE; break;
            }
        } while (k != null);
        jq_Class[] interfaces = klass.getDeclaredInterfaces();
        for (int i=0; i<interfaces.length; ++i) {
            jq_Class k2 = interfaces[i];
            byte res2 = implementsInterface_noload(k2, inter);
            if (res2 == YES) return YES;
            if (res2 == MAYBE) res = MAYBE;
        }
        return res;
    }

    static byte declaresInterface(jq_Class klass, Set interfaces, boolean loadClasses) {
        jq.assert(klass.isLoaded());
        jq_Class[] klass_interfaces = klass.getDeclaredInterfaces();
        for (int i=0; i<klass_interfaces.length; ++i) {
            if (!loadClasses && !klass_interfaces[i].isLoaded()) return MAYBE;
            if (interfaces.contains(klass_interfaces[i])) return YES;
        }
        return NO;
    }
    
    public static CallTargets getTargets(jq_Class callingClass,
                                         jq_Method method,
                                         byte type,
                                         boolean loadClasses)
    {
        if (type == BytecodeVisitor.INVOKE_STATIC) return getStaticTargets((jq_StaticMethod)method);
        jq_InstanceMethod imethod = (jq_InstanceMethod)method;
        if (type == BytecodeVisitor.INVOKE_SPECIAL) return getSpecialTargets(callingClass, imethod, loadClasses);
        if (type == BytecodeVisitor.INVOKE_VIRTUAL)
            return getTargets(callingClass, imethod, type, imethod.getDeclaringClass(), false, loadClasses);
        jq.assert(type == BytecodeVisitor.INVOKE_INTERFACE);
        jq.assert(!imethod.getDeclaringClass().isLoaded() ||
                  imethod.getDeclaringClass().isInterface());

        // find the set of equivalent interfaces
        jq_Class interf = method.getDeclaringClass();
        Set interfaces = new LinearSet(); interfaces.add(interf);
        boolean again = false;
        do {
            jq_Class rclass = PrimordialClassLoader.loader.getJavaLangObject();
            Stack subclass = new Stack();
            subclass.push(rclass);
            while (!subclass.empty()) {
                rclass = (jq_Class)subclass.pop();
                if (!rclass.isLoaded()) {
                    if (!loadClasses) {
                        continue; // conservative.
                    }
                    rclass.load();
                }
                if (!rclass.isInterface()) continue;
                if (declaresInterface(rclass, interfaces, loadClasses) != NO) {
                    again = true; // must repeat to catch any interfaces that implement this one
                    // add subtree from here.
                    Stack subclass2 = new Stack();
                    subclass2.add(rclass);
                    for (;;) {
                        rclass = (jq_Class)subclass2.pop();
                        if (!rclass.isLoaded()) {
                            if (!loadClasses) {
                                continue; // conservative.
                            }
                            rclass.load();
                        }
                        interfaces.add(rclass);
                        jq_Class[] subclasses = rclass.getSubClasses();
                        for (int i=0; i<subclasses.length; ++i) subclass2.push(subclasses[i]);
                    }
                }
            }
        } while (again);
        
        // find the set of classes that implement these interfaces.
        Stack subclass1 = new Stack(); // don't implement
        Stack subclass2 = new Stack(); // do/may implement
        jq_Class rclass = PrimordialClassLoader.loader.getJavaLangObject();
        if (rclass.implementsInterface(interf)) subclass2.push(rclass);
        else subclass1.push(rclass);
        while (!subclass1.empty()) {
            rclass = (jq_Class)subclass1.pop();
            if (!rclass.isLoaded()) {
                if (!loadClasses) {
                    continue; // conservative.
                }
                rclass.load();
            }
            if (rclass.isInterface()) continue;
            if (declaresInterface(rclass, interfaces, loadClasses) != NO) {
                subclass2.push(rclass);
            }
        }
        Set implementors = new HashSet();
        
        Set c = new HashSet(); // use a HashSet because it is going to be large
        while (!subclass2.empty()) {
            rclass = (jq_Class)subclass2.pop();
            if (!rclass.isLoaded()) {
                if (!loadClasses) {
                    continue; // conservative.
                }
                rclass.load();
            }
            jq.assert(!rclass.isInterface());
            jq_InstanceMethod target = (jq_InstanceMethod)rclass.getDeclaredMember(imethod.getNameAndDesc());
            if (target != null) {
                if (!target.isAbstract()) c.add(target);
                if (!target.isFinal() && !target.isPrivate()) {
                    jq_Class[] subclasses = rclass.getSubClasses();
                    for (int i=0; i<subclasses.length; ++i) subclass2.push(subclasses[i]);
                }
            }
        }
        return new MultipleCallTargets(c, false);
    }

    public static SingleCallTarget getStaticTargets(jq_StaticMethod method)
    {
        // static method. the declaring class might not have been loaded.
        return new SingleCallTarget(method, true);
    }

    public static CallTargets getSpecialTargets(jq_Class callingClass,
                                                jq_InstanceMethod method,
                                                boolean loadClasses)
    {
        // special, non-virtual invocation.
        if (!method.getDeclaringClass().isLoaded()) {
            if (callingClass.isLoaded() && !callingClass.isSpecial())
                return new SingleCallTarget(method, true);
            if (method.isInitializer()) return new SingleCallTarget(method, true);
            if (!loadClasses) return NoCallTarget.INSTANCE;  // no idea!
            method.getDeclaringClass().load();
        }
        jq_InstanceMethod target = jq_Class.getInvokespecialTarget(callingClass, method);
        return new SingleCallTarget(target, true);
    }
    
    public abstract Iterator iterator();
    public abstract boolean isComplete();
    public abstract CallTargets union(CallTargets s);
    
    static class NoCallTarget extends CallTargets
    {
        public Iterator iterator() { return NullIterator.INSTANCE; }
        public boolean isComplete() { return false; }
        public CallTargets union(CallTargets s) { return s; }
        public static final NoCallTarget INSTANCE = new NoCallTarget();
        private NoCallTarget() {}
        public String toString() { return "{}"; }
    }
    
    static class SingleCallTarget extends CallTargets
    {
        final jq_Method method; final boolean complete;
        SingleCallTarget(jq_Method m, boolean c) { method = m; complete = c; }
        public Iterator iterator() { return new SingletonIterator(method); }
        public boolean isComplete() { return complete; }
        public CallTargets union(CallTargets s) {
            if (s == NoCallTarget.INSTANCE) return this;
            Set result = new LinearSet();
            boolean is_complete = this.complete;
            result.add(this.method);
            if (!s.isComplete()) is_complete = false;
            for (Iterator i = s.iterator(); i.hasNext(); ) {
                result.add(i.next());
            }
            return new MultipleCallTargets(result, is_complete);
        }
        public String toString() {
            if (complete) return "{ "+method.toString()+" } (complete)";
            else return "{ "+method.toString()+" }";
        }
    }
    
    static class MultipleCallTargets extends CallTargets
    {
        final Set set; boolean complete;
        MultipleCallTargets(Set s, boolean c) { set = s; complete = c; }
        public Iterator iterator() { return set.iterator(); }
        public boolean isComplete() { return complete; }
        public CallTargets union(CallTargets s) {
            if (s == NoCallTarget.INSTANCE) return this;
            if (s instanceof SingleCallTarget) {
                SingleCallTarget sct = (SingleCallTarget)s;
                this.set.add(sct.method);
                if (!sct.isComplete()) this.complete = false;
            } else {
                jq.assert(s instanceof MultipleCallTargets);
                this.set.addAll(((MultipleCallTargets)s).set);
            }
            return this;
        }
        public String toString() {
            if (complete) return set.toString()+" (complete)";
            else return set.toString();
        }
    }
    
}
