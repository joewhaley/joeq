// List.java, created Wed Mar  5  0:26:32 2003 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Util.Templates;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public abstract class List {
    public interface jq_Type extends java.util.List {
        Clazz.jq_Type getType(int index);
        ListIterator.jq_Type typeIterator();
    }
    public interface jq_Reference extends jq_Type {
        Clazz.jq_Reference getReference(int index);
        ListIterator.jq_Reference referenceIterator();
    }
    public interface jq_Class extends jq_Reference {
        Clazz.jq_Class getClass(int index);
        ListIterator.jq_Class classIterator();
    }
    public interface jq_Member extends java.util.List {
        Clazz.jq_Member getMember(int index);
        ListIterator.jq_Member memberIterator();
    }
    public interface jq_Method extends jq_Member {
        Clazz.jq_Method getMethod(int index);
        ListIterator.jq_Method methodIterator();
    }
    public interface jq_InstanceMethod extends jq_Method {
        Clazz.jq_InstanceMethod getInstanceMethod(int index);
        ListIterator.jq_InstanceMethod instanceMethodIterator();
    }
    public interface jq_StaticMethod extends jq_Method {
        Clazz.jq_StaticMethod getStaticMethod(int index);
        ListIterator.jq_StaticMethod staticMethodIterator();
    }
        
    public interface BasicBlock extends java.util.List {
        Compil3r.Quad.BasicBlock getBasicBlock(int index);
        ListIterator.BasicBlock basicBlockIterator();
    }
    public interface ExceptionHandler extends java.util.List {
        Compil3r.Quad.ExceptionHandler getExceptionHandler(int index);
        ListIterator.ExceptionHandler exceptionHandlerIterator();
    }
    public interface Quad extends java.util.List {
        Compil3r.Quad.Quad getQuad(int index);
        ListIterator.Quad quadIterator();
    }
    public interface RegisterOperand extends java.util.List {
        Compil3r.Quad.Operand.RegisterOperand getRegisterOperand(int index);
        ListIterator.RegisterOperand registerOperandIterator();
    }
}
