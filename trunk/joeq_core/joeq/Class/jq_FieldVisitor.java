/*
 * jq_FieldVisitor.java
 *
 * Created on January 9, 2002, 9:40 AM
 *
 * @author  John Whaley
 * @version 
 */

package Clazz;

import Util.ArrayIterator;
import Util.AppendIterator;
import java.util.Iterator;

public interface jq_FieldVisitor {

    public void visitStaticField(jq_StaticField m);
    public void visitInstanceField(jq_InstanceField m);
    public void visitField(jq_Field m);
    
    public class EmptyVisitor implements jq_FieldVisitor {
        public void visitStaticField(jq_StaticField m) {}
        public void visitInstanceField(jq_InstanceField m) {}
        public void visitField(jq_Field m) {}
    }
    
    public class DeclaredFieldVisitor extends jq_TypeVisitor.EmptyVisitor {
        final jq_FieldVisitor mv; boolean trace;
        public DeclaredFieldVisitor(jq_FieldVisitor mv) { this.mv = mv; }
        public DeclaredFieldVisitor(jq_FieldVisitor mv, boolean trace) { this.mv = mv; this.trace = trace; }
        public void visitClass(jq_Class k) {
            if (trace) System.out.println(k.toString());
            Iterator it = new AppendIterator(new ArrayIterator(k.getDeclaredStaticFields()),
                                                new ArrayIterator(k.getDeclaredInstanceFields()));
            while (it.hasNext()) {
                jq_Field m = (jq_Field)it.next();
                m.accept(mv);
            }
        }
    }
    
}

