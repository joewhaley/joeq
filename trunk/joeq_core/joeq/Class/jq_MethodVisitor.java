/*
 * jq_MethodVisitor.java
 *
 * Created on January 9, 2002, 9:40 AM
 *
 * @author  Administrator
 * @version 
 */

package Clazz;

import Util.ArrayIterator;
import Util.AppendIterator;
import java.util.Iterator;

public interface jq_MethodVisitor {

    public void visitClassInitializer(jq_ClassInitializer m);
    public void visitInitializer(jq_Initializer m);
    public void visitStaticMethod(jq_StaticMethod m);
    public void visitInstanceMethod(jq_InstanceMethod m);
    public void visitMethod(jq_Method m);
    
    public class EmptyVisitor implements jq_MethodVisitor {
        public void visitClassInitializer(jq_ClassInitializer m) {}
        public void visitInitializer(jq_Initializer m) {}
        public void visitStaticMethod(jq_StaticMethod m) {}
        public void visitInstanceMethod(jq_InstanceMethod m) {}
        public void visitMethod(jq_Method m) {}
    }
    
    public class DeclaredMethodVisitor extends jq_TypeVisitor.EmptyVisitor {
        final jq_MethodVisitor mv; boolean trace;
        public DeclaredMethodVisitor(jq_MethodVisitor mv) { this.mv = mv; }
        public DeclaredMethodVisitor(jq_MethodVisitor mv, boolean trace) { this.mv = mv; this.trace = trace; }
        public void visitClass(jq_Class k) {
            if (trace) System.out.println(k.toString());
            Iterator it = new AppendIterator(new ArrayIterator(k.getDeclaredStaticMethods()),
                                                new ArrayIterator(k.getDeclaredInstanceMethods()));
            while (it.hasNext()) {
                jq_Method m = (jq_Method)it.next();
                m.accept(mv);
            }
        }
    }
    
}

