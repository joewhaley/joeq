/*
 * jq_MethodVisitor.java
 *
 * Created on January 9, 2002, 9:40 AM
 *
 */

package Clazz;

/*
 * @author  John Whaley
 * @version $Id$
 */
public interface jq_TypeVisitor {

    void visitClass(jq_Class m);
    void visitArray(jq_Array m);
    void visitPrimitive(jq_Primitive m);
    void visitType(jq_Type m);
    
    class EmptyVisitor implements jq_TypeVisitor {
        public void visitClass(jq_Class m) {}
        public void visitArray(jq_Array m) {}
        public void visitPrimitive(jq_Primitive m) {}
        public void visitType(jq_Type m) {}
    }
    
}
