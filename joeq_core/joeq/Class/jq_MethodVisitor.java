/*
 * jq_MethodVisitor.java
 *
 * Created on January 9, 2002, 9:40 AM
 *
 */

package Clazz;

import java.util.Iterator;
import java.util.Set;

import Util.AppendIterator;
import Util.ArrayIterator;

/*
 * @author  John Whaley
 * @version $Id$
 */
public interface jq_MethodVisitor {

    void visitClassInitializer(jq_ClassInitializer m);
    void visitInitializer(jq_Initializer m);
    void visitStaticMethod(jq_StaticMethod m);
    void visitInstanceMethod(jq_InstanceMethod m);
    void visitMethod(jq_Method m);
    
    class EmptyVisitor implements jq_MethodVisitor {
        public void visitClassInitializer(jq_ClassInitializer m) {}
        public void visitInitializer(jq_Initializer m) {}
        public void visitStaticMethod(jq_StaticMethod m) {}
        public void visitInstanceMethod(jq_InstanceMethod m) {}
        public void visitMethod(jq_Method m) {}
    }
    
    class DeclaredMethodVisitor extends jq_TypeVisitor.EmptyVisitor {
        final jq_MethodVisitor mv; final Set methodNames; boolean trace;
        public DeclaredMethodVisitor(jq_MethodVisitor mv) { this.mv = mv; this.methodNames = null; this.trace = false; }
        public DeclaredMethodVisitor(jq_MethodVisitor mv, boolean trace) { this.mv = mv; this.methodNames = null; this.trace = trace; }
        public DeclaredMethodVisitor(jq_MethodVisitor mv, Set methodNames, boolean trace) { this.mv = mv; this.methodNames = methodNames; this.trace = trace; }
        public void visitClass(jq_Class k) {
            if (trace) System.out.println(k.toString());
            Iterator it = new AppendIterator(new ArrayIterator(k.getDeclaredStaticMethods()),
                                                new ArrayIterator(k.getDeclaredInstanceMethods()));
            while (it.hasNext()) {
                jq_Method m = (jq_Method)it.next();
                if (methodNames != null && !methodNames.contains(m.getName().toString()))
                    continue;
                m.accept(mv);
            }
        }
    }
    
}

