// Dominators.java, created Wed Jan 30 22:34:43 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Compil3r.Quad;
import java.util.ArrayList;
import java.util.Iterator;

import joeq.Clazz.jq_Method;
import joeq.Clazz.jq_MethodVisitor;
import joeq.Util.BitString;
import joeq.Util.Templates.List;
import joeq.Util.Templates.ListIterator;

/**
 *
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class Dominators extends jq_MethodVisitor.EmptyVisitor implements BasicBlockVisitor {

    /** true = normal dominators.
     * false = post dominators.
     */
    public Dominators(boolean direction) {
        this.direction = direction;
    }
    public Dominators() {
        this(false);
    }

    public static final boolean TRACE = false;

    public final boolean direction;
    public BitString[] dominators;
    protected boolean change;
    protected ControlFlowGraph cfg;
    protected BasicBlock[] bbs;
    private BitString temp;
    
    public void visitMethod(jq_Method m) {
        if (m.getBytecode() == null) return;
        cfg = Compil3r.Quad.CodeCache.getCode(m);
        bbs = new BasicBlock[cfg.getNumberOfBasicBlocks()];
        dominators = new BitString[cfg.getNumberOfBasicBlocks()];
        temp = new BitString(dominators.length);
        int offset = direction?1:0;
        dominators[offset] = new BitString(dominators.length);
        dominators[offset].setAll();
        dominators[1-offset] = new BitString(dominators.length);
        dominators[1-offset].set(1-offset);
        for (int i=2; i<dominators.length; ++i) {
            dominators[i] = new BitString(dominators.length);
            dominators[i].setAll();
        }
        List.BasicBlock rpo;
        if (direction)
            rpo = cfg.reversePostOrder(cfg.entry());
        else
            rpo = cfg.reversePostOrderOnReverseGraph(cfg.exit());
        for (;;) {
            if (TRACE) System.out.println("Iterating over "+rpo);
            change = false;
            ListIterator.BasicBlock rpo_i = rpo.basicBlockIterator();
            BasicBlock first = rpo_i.nextBasicBlock(); // skip first node.
            bbs[first.getID()] = first;
            while (rpo_i.hasNext()) {
                BasicBlock bb = rpo_i.nextBasicBlock();
                this.visitBasicBlock(bb);
            }
            if (!change) break;
        }
        /*for (int i=0; i<dominators.length; ++i) {
            System.out.println("Dom "+i+": "+dominators[i]);
        }*/
        //computeTree();
        
    }

    public void visitBasicBlock(BasicBlock bb) {
        if (TRACE) System.out.println("Visiting: "+bb);
        bbs[bb.getID()] = bb;
        temp.setAll();
        ListIterator.BasicBlock preds = direction?bb.getPredecessors().basicBlockIterator():
                                                  bb.getSuccessors().basicBlockIterator();
        while (preds.hasNext()) {
            BasicBlock pred = preds.nextBasicBlock();
            if (TRACE) System.out.println("Visiting pred: "+pred);
            temp.and(dominators[pred.getID()]);
        }
        if (direction) {
            if (bb.isExceptionHandlerEntry()) {
                Iterator it = cfg.getExceptionHandlersMatchingEntry(bb);
                while (it.hasNext()) {
                    ExceptionHandler eh = (ExceptionHandler)it.next();
                    preds = eh.getHandledBasicBlocks().basicBlockIterator();
                    while (preds.hasNext()) {
                        BasicBlock pred = preds.nextBasicBlock();
                        if (TRACE) System.out.println("Visiting ex pred: "+pred);
                        temp.and(dominators[pred.getID()]);
                    }
                }
            }
        } else {
            ListIterator.ExceptionHandler it = bb.getExceptionHandlers().exceptionHandlerIterator();
            while (it.hasNext()) {
                ExceptionHandler eh = (ExceptionHandler)it.next();
                BasicBlock pred = eh.getEntry();
                if (TRACE) System.out.println("Visiting ex pred: "+pred);
                temp.and(dominators[pred.getID()]);
            }
        }
        temp.set(bb.getID());
        if (!temp.equals(dominators[bb.getID()])) {
            if (TRACE) System.out.println("Changed!");
            //dominators[bb.getID()] <- temp
            dominators[bb.getID()].copyBits(temp);
            change = true;
        }
        //reset change to break the loop
        //else change = false; 
    }
    
    public DominatorNode computeTree() {
        // TODO: fix this. this algorithm sucks (n^4 or so)
        ArrayList list = new ArrayList();
        list.add(new ArrayList());
        for (int depth = 1; ; ++depth) {
            if (TRACE) System.out.println("depth: "+depth);
            ArrayList list2 = new ArrayList();
            boolean found = false;
            for (int i=0; i<dominators.length; ++i) {
                if (dominators[i].numberOfOnes() == depth) {
                    if (TRACE) System.out.println("bb"+i+" matches: "+dominators[i]);
                    found = true;
                    temp.copyBits(dominators[i]);
                    temp.clear(i);
                    DominatorNode parent = null;
                    Iterator it = ((ArrayList)list.get(depth-1)).iterator();
                    while (it.hasNext()) {
                        DominatorNode n = (DominatorNode)it.next();
                        if (temp.equals(dominators[n.getBasicBlock().getID()])) {
                            parent = n; break;
                        }
                    }
                    DominatorNode n0 = new DominatorNode(bbs[i], parent);
                    if (parent != null)
                        parent.addChild(n0);
                    list2.add(n0);
                }
            }
            list.add(list2);
            if (!found) break;
        }
        DominatorNode root = (DominatorNode)((ArrayList)list.get(1)).get(0);
        return root;
    }
    
    public static class DominatorNode {
        public final BasicBlock bb;
        public final DominatorNode parent;
        public final ArrayList children;
        
        public DominatorNode(BasicBlock bb, DominatorNode parent) {
            this.bb = bb; this.parent = parent; this.children = new ArrayList();
        }
        
        public BasicBlock getBasicBlock() { return bb; }
        public DominatorNode getParent() { return parent; }
        public int getNumberOfChildren() { return children.size(); }
        public DominatorNode getChild(int i) { return (DominatorNode)children.get(i); }
        public java.util.List getChildren() { return children; }
        public void addChild(DominatorNode n) { children.add(n); }
        public String toString() { return bb.toString(); }
        public void dumpTree() {
            System.out.println("Node: "+toString());
            System.out.println("Children of :"+toString());
            Iterator i = children.iterator();
            while (i.hasNext()) {
                ((DominatorNode)i.next()).dumpTree();
            }
            System.out.println("End of children of :"+toString());
        }
    }
    
    public static void main(String[] args) {
        Main.HostedVM.initialize();
        
        Clazz.jq_Class c = (Clazz.jq_Class) Clazz.jq_Type.parseType(args[0]);
        c.load();
        Dominators dom = new Dominators();
        jq_Method[] ms = c.getDeclaredStaticMethods();
        for (int i=0; i<ms.length; ++i) {
            dom.visitMethod(ms[i]);
            DominatorNode n = dom.computeTree();
            n.dumpTree();
        }
    }
    
}
