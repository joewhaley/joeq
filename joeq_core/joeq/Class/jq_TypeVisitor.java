/*
 * jq_MethodVisitor.java
 *
 * Created on January 9, 2002, 9:40 AM
 *
 * @author  Administrator
 * @version 
 */

package Clazz;

public interface jq_TypeVisitor {

    public void visitClass(jq_Class m);
    public void visitArray(jq_Array m);
    public void visitPrimitive(jq_Primitive m);
    public void visitType(jq_Type m);
    
    public class EmptyVisitor implements jq_TypeVisitor {
        public void visitClass(jq_Class m) {}
        public void visitArray(jq_Array m) {}
        public void visitPrimitive(jq_Primitive m) {}
        public void visitType(jq_Type m) {}
    }
    
}
