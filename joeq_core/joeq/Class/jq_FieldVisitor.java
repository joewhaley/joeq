/*
 * jq_FieldVisitor.java
 *
 * Created on January 9, 2002, 9:40 AM
 *
 */

package Clazz;

import java.util.Arrays;
import java.util.Iterator;

import Util.Collections.AppendIterator;

/*
 * @author  John Whaley
 * @version $Id$
 */
public interface jq_FieldVisitor {

    void visitStaticField(jq_StaticField m);
    void visitInstanceField(jq_InstanceField m);
    void visitField(jq_Field m);
    
    class EmptyVisitor implements jq_FieldVisitor {
        public void visitStaticField(jq_StaticField m) {}
        public void visitInstanceField(jq_InstanceField m) {}
        public void visitField(jq_Field m) {}
    }
    
    class DeclaredFieldVisitor extends jq_TypeVisitor.EmptyVisitor {
        final jq_FieldVisitor mv; boolean trace;
        public DeclaredFieldVisitor(jq_FieldVisitor mv) { this.mv = mv; }
        public DeclaredFieldVisitor(jq_FieldVisitor mv, boolean trace) { this.mv = mv; this.trace = trace; }
        public void visitClass(jq_Class k) {
            if (trace) System.out.println(k.toString());
            Iterator it = new AppendIterator(Arrays.asList(k.getDeclaredStaticFields()).iterator(),
                                            Arrays.asList(k.getDeclaredInstanceFields()).iterator());
            while (it.hasNext()) {
                jq_Field m = (jq_Field)it.next();
                m.accept(mv);
            }
        }
    }
    
}

