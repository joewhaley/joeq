// ListIterator.java, created Wed Mar  5  0:26:32 2003 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.Templates;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public abstract class ListIterator {
    public interface jq_Type extends java.util.ListIterator {
        joeq.Clazz.jq_Type nextType();
        joeq.Clazz.jq_Type previousType();
    }
    public interface jq_Reference extends jq_Type {
        joeq.Clazz.jq_Reference nextReference();
        joeq.Clazz.jq_Reference previousReference();
    }
    public interface jq_Class extends jq_Reference {
        joeq.Clazz.jq_Class nextClass();
        joeq.Clazz.jq_Class previousClass();
    }
    public interface jq_Member extends java.util.ListIterator {
        joeq.Clazz.jq_Member nextMember();
        joeq.Clazz.jq_Member previousMember();
    }
    public interface jq_Method extends jq_Member {
        joeq.Clazz.jq_Method nextMethod();
        joeq.Clazz.jq_Method previousMethod();
    }
    public interface jq_InstanceMethod extends jq_Method {
        joeq.Clazz.jq_InstanceMethod nextInstanceMethod();
        joeq.Clazz.jq_InstanceMethod previousInstanceMethod();
    }
    public interface jq_StaticMethod extends jq_Method {
        joeq.Clazz.jq_StaticMethod nextStaticMethod();
        joeq.Clazz.jq_StaticMethod previousStaticMethod();
    }
        
    public interface BasicBlock extends java.util.ListIterator {
        joeq.Compil3r.Quad.BasicBlock nextBasicBlock();
        joeq.Compil3r.Quad.BasicBlock previousBasicBlock();
    }
    public interface ExceptionHandler extends java.util.ListIterator {
        joeq.Compil3r.Quad.ExceptionHandler nextExceptionHandler();
        joeq.Compil3r.Quad.ExceptionHandler previousExceptionHandler();
    }
    public interface Quad extends java.util.ListIterator {
        joeq.Compil3r.Quad.Quad nextQuad();
        joeq.Compil3r.Quad.Quad previousQuad();
    }
    public interface Operand extends java.util.ListIterator {
        joeq.Compil3r.Quad.Operand nextOperand();
        joeq.Compil3r.Quad.Operand previousOperand();
    }
    public interface RegisterOperand extends Operand {
        joeq.Compil3r.Quad.Operand.RegisterOperand nextRegisterOperand();
        joeq.Compil3r.Quad.Operand.RegisterOperand previousRegisterOperand();
    }
}
