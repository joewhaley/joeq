// TypeAnalysis.java, created Fri Jan 11 16:49:00 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Compil3r.BytecodeAnalysis;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.Stack;

import joeq.Bootstrap.PrimordialClassLoader;
import joeq.Clazz.jq_Array;
import joeq.Clazz.jq_Class;
import joeq.Clazz.jq_Field;
import joeq.Clazz.jq_InstanceField;
import joeq.Clazz.jq_InstanceMethod;
import joeq.Clazz.jq_Method;
import joeq.Clazz.jq_Reference;
import joeq.Clazz.jq_StaticField;
import joeq.Clazz.jq_StaticMethod;
import joeq.Clazz.jq_Type;
import joeq.Run_Time.Reflection;
import joeq.Run_Time.TypeCheck;
import joeq.UTF.Utf8;
import joeq.Util.Assert;
import joeq.Util.Strings;
import joeq.Util.Collections.IdentityHashCodeWrapper;
import joeq.Util.Collections.LinearSet;

/**
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class TypeAnalysis {
    
    static Map/*<jq_Method, AnalysisSummary>*/ summaries = new HashMap();

    public static int nBytesAnalyzed;
    public static int nMethods;

    public static void dump() throws java.io.IOException {
        out_ta.println("Total number of methods analyzed: "+nMethods);
        out_ta.println("Total number of bytes of code analyzed: "+nBytesAnalyzed);
        if (MethodCallSequence.modeler != null) MethodCallSequence.modeler.dump();
    }
    
    public static Set classesToAnalyze;
    
    public static AnalysisSummary analyze(jq_Method m) {
        AnalysisSummary r = (AnalysisSummary)summaries.get(m);
        if (r != null) return r;
        return analyze(m, new Stack(), new LinearSet());
    }
    
    public static boolean TRACE_MAIN = true;
    public static boolean TRACE_ITERATION = false;
    public static boolean DUMP_SUMMARY = false;
    public static final java.io.PrintStream out_ta = System.out;
    
    static AnalysisSummary analyze(jq_Method m, Stack callStack, Set do_it_again) {
        if (TRACE_MAIN) out_ta.println("Analyzing "+m+" depth "+callStack.size());
        nBytesAnalyzed += m.getBytecode().length;
        ++nMethods;
        callStack.push(m);
        //  --- compute cfg for method
        ControlFlowGraph cfg = ControlFlowGraph.computeCFG(m);
        //  --- initialize bb in states
        AnalysisState[] in_states = new AnalysisState[cfg.getNumberOfBasicBlocks()];
        AnalysisState[] out_states = new AnalysisState[cfg.getNumberOfBasicBlocks()];
        in_states[2] = AnalysisState.makeEntry(m);
        //// in_states[1] is the normal exit
        //// in_states[0] is the exceptional exit
        //  --- initialize visitor
        TypeAnalysisVisitor tav = new TypeAnalysisVisitor(m, cfg, callStack, do_it_again, in_states, out_states);
        //  --- initialize stack depths
        for (int i=0; i<cfg.getNumberOfBasicBlocks(); ++i) {
            BasicBlock bb = cfg.getBasicBlock(i);
            bb.startingStackDepth = Integer.MAX_VALUE;
        }
        cfg.getBasicBlock(2).startingStackDepth = 0;
        //  --- iterate in RPO
        boolean change;
        do {
            if (TRACE_ITERATION) out_ta.println("Computing reverse post order");
            ControlFlowGraph.RPOBasicBlockIterator rpo = cfg.reversePostOrderIterator();
            BasicBlock first_bb = rpo.nextBB();
            Assert._assert(first_bb == cfg.getEntry());
            change = false;
            while (rpo.hasNext()) {
                BasicBlock bb = rpo.nextBB();
                Assert._assert(bb.id != 0);
                if (bb.id == 1) continue;
                if (in_states[bb.id] == null) {
                    if (TRACE_ITERATION) out_ta.println("Can't find in set for "+bb);
                    continue;
                }
                tav.currentState = in_states[bb.id].copy_deep();
                tav.change = false;
                if (TRACE_ITERATION) out_ta.println("Visiting basic block "+bb+" stack depth "+bb.startingStackDepth+" state=");
                if (TRACE_ITERATION) tav.currentState.dump();
                tav.currentStackDepth = bb.startingStackDepth;
                tav.dontMergeWithSuccessors = false;
                tav.visitBasicBlock(bb);
                if (TRACE_ITERATION && tav.change) out_ta.println("Change in sets detected within the basic block!");
                change |= tav.change;
                if (tav.dontMergeWithSuccessors) {
                    if (TRACE_ITERATION) out_ta.println("skipping merge with successors of "+bb);
                } else {
                    for (int i=0; i<bb.getNumberOfSuccessors(); ++i) {
                        BasicBlock bb2 = bb.getSuccessor(i);
                        if (in_states[bb2.id] != null) {
                            if (in_states[bb2.id].union_deep(tav.currentState)) {
                                if (TRACE_ITERATION) out_ta.println("In set for "+bb2+" changed!");
                                change = true;
                            } else {
                                if (TRACE_ITERATION) out_ta.println("In set for "+bb2+" did not change");
                            }
                            if (bb2.id == 1) {
                                Assert._assert(tav.currentStackDepth == 0);
                                bb2.startingStackDepth = 0;
                            } else {
                                Assert._assert(bb2.startingStackDepth == tav.currentStackDepth,
                                          "Stack depth mismatch: "+bb2+"="+bb2.startingStackDepth+", "+tav+"="+tav.currentStackDepth);
                            }
                        } else {
                            if (TRACE_ITERATION) out_ta.println("No in set for "+bb2+" yet");
                            in_states[bb2.id] = tav.currentState.copy_deep();
                            change = true;
                            bb2.startingStackDepth = tav.currentStackDepth;
                        }
                    }
                }
                out_states[bb.id] = tav.currentState;
            }
        } while (change);
        //  --- build summary
        if (TRACE_ITERATION) out_ta.println("Finished iteration of "+m+", building summary.");
        AnalysisSummary summary = tav.summary;
        summary.finish(in_states[1], in_states[0]);
        Object o = callStack.pop();
        Assert._assert(o == m);
        if (TRACE_MAIN) out_ta.println("Finished "+m+" depth "+callStack.size());
        if (DUMP_SUMMARY) summary.dump(false);
        return summary;
    }
    
    static class SetOfLocations {
        
        public static final boolean TRACE_SET_CREATION = false;
        public static final boolean TRACE_DEREFERENCES = false;
        public static final boolean TRACE_FILTER = false;
        public static final boolean TRACE_CALLHISTORY = false;
        public static final boolean TRACE_UNION = false;
        
        // 1:1, this field is owned by this object
        private final LinearSet/*<ProgramLocation>*/ sources;
        
        public String toString() {
            return sources+"@"+Integer.toHexString(this.hashCode());
        }
        
        private SetOfLocations() { sources = new LinearSet(); }
        
        static SetOfLocations makeParamSet(ParamLocation pl) {
            SetOfLocations t = new SetOfLocations();
            t.sources.add(pl);
            if (TRACE_SET_CREATION) out_ta.println("Made "+t);
            return t;
        }
        
        static SetOfLocations makeStaticFieldSet(jq_StaticField f) {
            SetOfLocations t = new SetOfLocations();
            StaticFieldLocation sfl = new StaticFieldLocation(f, f.getType());
            sfl.methodSequences = new MethodCallSequences(true);
            t.sources.add(sfl);
            if (TRACE_SET_CREATION) out_ta.println("Made "+t);
            return t;
        }
        
        static SetOfLocations makeDerefSet(DereferenceLocation d) {
            SetOfLocations t = new SetOfLocations();
            t.sources.add(d);
            if (TRACE_SET_CREATION) out_ta.println("Made deref set "+t);
            return t;
        }
        
        static SetOfLocations makeNewSet(jq_Type createdType, jq_Method m, int bcindex) {
            SetOfLocations t = new SetOfLocations();
            AllocationLocation loc = new AllocationLocation(createdType, m, bcindex);
            loc.methodSequences = new MethodCallSequences(false);
            t.sources.add(loc);
            if (TRACE_SET_CREATION) out_ta.println("Made "+t);
            return t;
        }
        
        static SetOfLocations makeConstantSet(Object o, jq_Type createdType, jq_Method m, int bcindex) {
            SetOfLocations t = new SetOfLocations();
            LoadConstantLocation loc = new LoadConstantLocation(o, createdType, m, bcindex);
            loc.methodSequences = new MethodCallSequences(true);
            t.sources.add(loc);
            if (TRACE_SET_CREATION) out_ta.println("Made "+t);
            return t;
        }
        
        static SetOfLocations makeJSRSubroutine(BasicBlock bb) {
            SetOfLocations t = new SetOfLocations();
            JSRSubroutine loc = new JSRSubroutine(bb);
            t.sources.add(loc);
            if (TRACE_SET_CREATION) out_ta.println("Made jsr subroutine set "+t);
            return t;
        }
        
        static SetOfLocations makeEmptySet() {
            SetOfLocations t = new SetOfLocations();
            if (TRACE_SET_CREATION) out_ta.println("Made empty set "+t);
            return t;
        }
        
        static SetOfLocations makeSeed(jq_Type returnedType) {
            SetOfLocations t = new SetOfLocations();
            if (TRACE_SET_CREATION) out_ta.println("Made seed "+t);
            return t;
        }
        
        public static final boolean MERGE_DEREFS = true;

        SetOfLocations dereference(Dereference deref, jq_Method m, int bcindex) {
            if (MERGE_DEREFS) {
                Iterator i = sources.iterator();
                while (i.hasNext()) {
                    ProgramLocation pl = (ProgramLocation)i.next();
                    if (pl instanceof OutsideProgramLocation) {
                        OutsideProgramLocation opl = (OutsideProgramLocation)pl;
                        SetOfLocations deref_set = opl.getOutsideEdge(deref);
                        if (deref_set == null) continue;
                        if (deref_set.size() == 0) continue;
                        DereferenceLocation deref_loc = (DereferenceLocation)deref_set.iterator().next();
                        if (TRACE_DEREFERENCES) out_ta.println("Reusing deref loc "+deref_loc);
                        return dereference(deref_loc);
                    }
                }
                if (TRACE_DEREFERENCES) out_ta.println("No useful deref loc found, creating one.");
            }

            DereferenceLocation deref_loc = new DereferenceLocation(deref, m, bcindex, deref.getType());
            deref_loc.methodSequences = new MethodCallSequences(true);
            return dereference(deref_loc);
        }
        
        SetOfLocations dereference(DereferenceLocation deref_loc) {
            Dereference deref = deref_loc.deref;
            SetOfLocations result = new SetOfLocations();
            boolean from_outside = false;
            for (Iterator i=sources.iterator(); i.hasNext(); ) {
                ProgramLocation pl = (ProgramLocation)i.next();
                if (pl.isFromOutside()) {
                    from_outside = true;
                    ((OutsideProgramLocation)pl).addOutsideEdge(deref_loc);
                }
                result.addAll(pl.dereference(deref));
            }
            if (from_outside) {
                result.add(deref_loc);
            }
            if (TRACE_SET_CREATION) out_ta.println("Made deref return set "+result);
            if (TRACE_DEREFERENCES) out_ta.println("Dereference: "+this+"."+deref+" == "+result);
            return result;
        }
        
        SetOfLocations dereference(Dereference deref, SetOfLocations outside_nodes, HashMap old_to_new) {
            SetOfLocations result = new SetOfLocations();
            boolean from_outside = false;
            for (Iterator i=sources.iterator(); i.hasNext(); ) {
                ProgramLocation pl = (ProgramLocation)i.next();
                if (pl.isFromOutside()) {
                    from_outside = true;
                    for (Iterator j = outside_nodes.iterator(); j.hasNext(); ) {
                        DereferenceLocation deref_loc = (DereferenceLocation)j.next();
                        DereferenceLocation deref_copy = (DereferenceLocation)deref_loc.copy_shallow(old_to_new);
                        ((OutsideProgramLocation)pl).addOutsideEdge(deref_copy);
                    }
                }
                result.addAll(pl.dereference(deref));
            }
            if (from_outside) {
                for (Iterator j = outside_nodes.iterator(); j.hasNext(); ) {
                    DereferenceLocation deref_loc = (DereferenceLocation)j.next();
                    DereferenceLocation deref_copy = (DereferenceLocation)deref_loc.copy_shallow(old_to_new);
                    result.add(deref_copy);
                }
            }
            if (TRACE_SET_CREATION) out_ta.println("Made deref return set "+result);
            if (TRACE_DEREFERENCES) out_ta.println("Dereference: "+this+"."+deref+" == "+result);
            return result;
        }
        
        void store(Dereference deref, SetOfLocations t) {
            for (Iterator i=sources.iterator(); i.hasNext(); ) {
                ProgramLocation pl = (ProgramLocation)i.next();
                if (TRACE_DEREFERENCES) out_ta.println("Store: "+pl+"."+deref+" <- "+t);
                pl.store_weak(deref, t);
            }
        }
        
        // Shallow add
        boolean addAll(SetOfLocations that) {
            if (that != null)
                return this.sources.addAll(that.sources);
            return false;
        }
        void add(ProgramLocation pl) {
            this.sources.add(pl);
        }
        
        SetOfLocations filterByType(jq_Type t) {
            if (TRACE_FILTER) out_ta.println("Filtering "+this+" by type "+t);
            SetOfLocations that = new SetOfLocations();
            for (Iterator i=sources.iterator(); i.hasNext(); ) {
                ProgramLocation pl = (ProgramLocation)i.next();
                ProgramLocation pl2 = pl.filterByType(t);
                if (pl2 != null) that.sources.add(pl2);
            }
            if (TRACE_SET_CREATION) out_ta.println("Made filtered set "+that);
            return that;
        }
        
        SetOfLocations copy_shallow() {
            SetOfLocations s = new SetOfLocations();
            s.sources.addAll(this.sources);
            if (TRACE_SET_CREATION) out_ta.println("Made shallow-copied set "+s);
            return s;
        }
        
        SetOfLocations copy_deep(HashMap old_to_new) {
            SetOfLocations s = (SetOfLocations)old_to_new.get(IdentityHashCodeWrapper.create(this));
            if (s == null) {
                s = new SetOfLocations();
                old_to_new.put(IdentityHashCodeWrapper.create(this), s);
                Iterator i = this.sources.iterator();
                while (i.hasNext()) {
                    ProgramLocation pl = (ProgramLocation)i.next();
                    s.sources.add(pl.copy_deep(old_to_new));
                }
                if (TRACE_SET_CREATION) out_ta.println("Made deep-copied set "+s);
            }
            return s;
        }
        
        /*
        void mustCall(Set m, String loc) {
            if (m.isEmpty()) return;
            for (Iterator i=sources.iterator(); i.hasNext(); ) {
                ProgramLocation pl = (ProgramLocation)i.next();
                if (TRACE_CALLHISTORY) out_ta.println(pl+" must call "+m);
                pl.mustCall(m, loc);
            }
        }
         */
        
        void mustCall(MethodCall m, String loc) {
            for (Iterator i=sources.iterator(); i.hasNext(); ) {
                ProgramLocation pl = (ProgramLocation)i.next();
                if (TRACE_CALLHISTORY) out_ta.println(pl+" must call "+m.getName());
                pl.mustCall(m, loc);
            }
        }
        
        /*
        void mayCall(Set m, String loc) {
            if (m.isEmpty()) return;
            for (Iterator i=sources.iterator(); i.hasNext(); ) {
                ProgramLocation pl = (ProgramLocation)i.next();
                if (TRACE_CALLHISTORY) out_ta.println(pl+" may call "+m);
                pl.mayCall(m, loc);
            }
        }
         */
        
        void mayCall(MethodCall m, String loc) {
            for (Iterator i=sources.iterator(); i.hasNext(); ) {
                ProgramLocation pl = (ProgramLocation)i.next();
                if (TRACE_CALLHISTORY) out_ta.println(pl+" may call "+m.getName());
                pl.mayCall(m, loc);
            }
        }
        
        // Changes "this", but not "that"
        boolean union_deep(SetOfLocations that, HashMap old_to_new, Stack stack) {
            if (this == that) return false;
            if (stack.contains(this)) {
                if (TRACE_UNION) out_ta.println("Cycle in set "+this+"!");
                return false;
            }
            stack.push(this);
            if (TRACE_UNION) out_ta.println("Unioning "+this+" and "+that);
            boolean change = false;
            for (Iterator i = that.sources.iterator(); i.hasNext(); ) {
                ProgramLocation that_pl = (ProgramLocation)i.next();
                ProgramLocation this_pl = (ProgramLocation)this.sources.get(that_pl);
                if (this_pl == null) {
                    if (TRACE_UNION) out_ta.println("New source: "+that_pl);
                    this.sources.add(this_pl = that_pl.copy_deep(old_to_new));
                    change = true;
                } else {
                    if (this_pl.union_deep(that_pl, old_to_new, stack)) change = true;
                }
            }
            Object o = stack.pop();
            Assert._assert(o == this);
            return change;
        }
        
        jq_Type getType() {
            Iterator i = this.sources.iterator();
            if (!i.hasNext()) return null;
            ProgramLocation pl = (ProgramLocation)i.next();
            jq_Type t = pl.getType();
            while (i.hasNext()) {
                pl = (ProgramLocation)i.next();
                jq_Type t2 = pl.getType();
                if (t == null) t = t2;
                if (t2 == null) continue;
                t = TypeCheck.findCommonSuperclass(t, t2, true);
            }
            return t;
        }
        
        boolean containsOutsideNode() {
            Iterator i = this.sources.iterator();
            while (i.hasNext()) {
                ProgramLocation pl = (ProgramLocation)i.next();
                if (pl.isFromOutside()) return true;
            }
            return false;
        }
        
        Iterator iterator() { return sources.iterator(); }
        int size() { return sources.size(); }
        boolean contains(ProgramLocation p) { return sources.contains(p); }
    }
    
    abstract static class Dereference {
        public abstract jq_Type getType();
    }
    // Immutable class
    static class ArrayDereference extends Dereference {
        private final jq_Array array_type;
        ArrayDereference(jq_Array type) { array_type = type; }
        public boolean equals(ArrayDereference that) {
            return (array_type == that.array_type);
            //return true;
        }
        public boolean equals(Object o) {
            if (o instanceof ArrayDereference) return equals((ArrayDereference)o);
            return false;
        }
        public int hashCode() {
            return array_type.hashCode();
            //return 1;
        }
        public String toString() {
            //return "("+array_type.getName()+")";
            return "[]";
        }
        public jq_Type getType() { return array_type.getElementType(); }
    }
    // Immutable class
    static class FieldDereference extends Dereference {
        private final jq_Field field;
        FieldDereference(jq_Field f) { field = f; }
        public boolean equals(FieldDereference that) {
            return (field == that.field);
        }
        public boolean equals(Object o) {
            if (o instanceof FieldDereference) return equals((FieldDereference)o);
            return false;
        }
        public int hashCode() { return field.hashCode(); }
        public String toString() { return field.getName().toString(); }
        public jq_Type getType() { return field.getType(); }
    }
    
    abstract static class ProgramLocation {
        protected Map/*<Dereference, SetOfLocations>*/ inside_edges;
        protected MethodCallSequences methodSequences;
        protected ProgramLocation() {
        }
        
        public static final boolean TRACE_UNION = false;
        public static final boolean TRACE_ADD_EDGE = false;
        public static final boolean TRACE_COPY = false;
        public static final boolean TRACE_FILTER = false;
        
        public abstract ProgramLocation copy_deep(HashMap old_to_new);
        
        protected void copyInsideEdges_deep(ProgramLocation that, HashMap old_to_new) {
            Assert._assert(that.inside_edges == null);
            if (this.inside_edges != null) {
                that.inside_edges = new HashMap();
                for (Iterator i=this.inside_edges.entrySet().iterator(); i.hasNext(); ) {
                    Map.Entry e = (Map.Entry)i.next();
                    SetOfLocations pl = (SetOfLocations)e.getValue();
                    that.inside_edges.put(e.getKey(), pl.copy_deep(old_to_new));
                }
            }
        }
        
        SetOfLocations dereference(Dereference d) {
            if (inside_edges != null) {
                return (SetOfLocations)inside_edges.get(d);
            } else {
                return null;
            }
        }
        
        SetOfLocations store_weak(Dereference d, SetOfLocations t) {
            SetOfLocations result = SetOfLocations.makeEmptySet();
            result.addAll(t);
            if (inside_edges == null) {
                inside_edges = new HashMap();
                inside_edges.put(d, result);
                if (TRACE_ADD_EDGE) out_ta.println("Adding first inside edge of "+this+d+" to "+result);
            } else {
                SetOfLocations s = (SetOfLocations)inside_edges.get(d);
                if (s != null) {
                    result.addAll(s); // weak update -> include all old edges too.
                    if (TRACE_ADD_EDGE) out_ta.println("Weak update on "+this+d+" to include "+t+", now set contains "+result);
                } else {
                    if (TRACE_ADD_EDGE) out_ta.println("Adding first inside edge on "+d+" to "+this+": "+result);
                }
                inside_edges.put(d, result);
            }
            return result;
        }
        
        // Changes "this", but not "that"
        boolean union_deep(ProgramLocation that, HashMap old_to_new, Stack stack) {
            if (this == that) return false;
            if (stack.contains(this)) {
                if (TRACE_UNION) out_ta.println("Cycle in location "+this+"!");
                return false;
            }
            stack.push(this);
            if (TRACE_UNION) out_ta.println("Unioning "+this+" and "+that);
            boolean result = false;
            if (this.methodSequences.union_deep(that.methodSequences)) result = true;
            if (this.unionInsideEdges_deep(that, old_to_new, stack)) result = true;
            Object o = stack.pop();
            Assert._assert(o == this);
            return result;
        }
        
        protected boolean unionInsideEdges_deep(ProgramLocation that, HashMap old_to_new, Stack stack) {
            if (this == that) return false;
            boolean change = false;
            if (that.inside_edges != null) {
                if (TRACE_UNION) out_ta.println("Unioning inside edges in "+this+" and "+that);
                if (this.inside_edges == null) this.inside_edges = new HashMap();
                for (Iterator i=that.inside_edges.entrySet().iterator(); i.hasNext(); ) {
                    Map.Entry e = (Map.Entry)i.next();
                    SetOfLocations this_s = (SetOfLocations)this.inside_edges.get(e.getKey());
                    SetOfLocations that_s = (SetOfLocations)e.getValue();
                    if (this_s == null) {
                        this_s = that_s.copy_deep(old_to_new);
                        this.inside_edges.put(e.getKey(), this_s);
                        change = true;
                    } else {
                        if (this_s.union_deep(that_s, old_to_new, stack)) change = true;
                    }
                }
            }
            return change;
        }
        
        public abstract ProgramLocation filterByType(jq_Type t);
        public abstract jq_Type getType();
        public boolean isFromOutside() { return false; }
        
        void mustCall(MethodCall m, String loc) {
            if (this.getType() == null) return;
            this.methodSequences.mustCall(loc, this.getType(), m);
        }
        /*
        void mustCall(Set m, String loc) {
            this.methodSequences.mustCall(loc, this.getType(), m);
        }
         */
        void mayCall(MethodCall m, String loc) {
            if (this.getType() == null) return;
            this.methodSequences.mayCall(loc, this.getType(), m);
        }
        /*
        void mayCall(Set m, String loc) {
            this.methodSequences.mayCall(loc, this.getType(), m);
        }
         */
        void addFirstCalledToMayCall(MethodCallSequences m, String loc) {
            if (this.getType() == null) return;
            this.methodSequences.addFirstCalledToMayCall(loc, this.getType(), m);
        }
        void updateMustCall(MethodCallSequences m) {
            if (this.getType() == null) return;
            this.methodSequences.updateMustCall(this.getType(), m);
        }
        
        public final Map getInsideEdges() { return inside_edges; }
        public final void initInsideEdges() { inside_edges = new HashMap(); }
        public final MethodCallSequences getMethodCallSequences() { return methodSequences; }
    }
    
    abstract static class OutsideProgramLocation extends ProgramLocation {
        private Map/*<Dereference, SetOfLocations*/ outside_edges;
        protected OutsideProgramLocation() {
        }
        public abstract jq_Type getOriginalType();
        public final Map getOutsideEdges() { return outside_edges; }
        public final void addOutsideEdge(DereferenceLocation d) {
            if (TRACE_ADD_EDGE) out_ta.println("Adding outside edge: "+this+"="+d.deref+"=>"+d);
            if (outside_edges == null) outside_edges = new HashMap();
            SetOfLocations s = (SetOfLocations)outside_edges.get(d.deref);
            if (s == null) {
                if (TRACE_ADD_EDGE) out_ta.println("First outside edge of deref "+d.deref);
                outside_edges.put(d.deref, SetOfLocations.makeDerefSet(d));
            } else {
                // not the only outside edge on this field!
                if (TRACE_ADD_EDGE) out_ta.println("Not the first outside edge of deref "+d.deref+"! Others are "+s);
                s.add(d);
            }
        }
        public final SetOfLocations getOutsideEdge(Dereference d) {
            if (outside_edges == null) return null;
            return (SetOfLocations) outside_edges.get(d);
        }
        protected void copyOutsideEdges_deep(OutsideProgramLocation that, HashMap old_to_new) {
            Assert._assert(that.outside_edges == null);
            if (this.outside_edges != null) {
                that.outside_edges = new HashMap();
                for (Iterator i=this.outside_edges.entrySet().iterator(); i.hasNext(); ) {
                    Map.Entry e = (Map.Entry)i.next();
                    SetOfLocations s = (SetOfLocations)e.getValue();
                    that.outside_edges.put(e.getKey(), s.copy_deep(old_to_new));
                }
            }
        }
        protected final boolean unionOutsideEdges_deep(OutsideProgramLocation that, HashMap old_to_new, Stack stack) {
            if (this == that) return false;
            boolean change = false;
            if (that.outside_edges != null) {
                if (TRACE_UNION) out_ta.println("Unioning outside edges in "+this+" and "+that);
                if (this.outside_edges == null) this.outside_edges = new HashMap();
                for (Iterator i=that.outside_edges.entrySet().iterator(); i.hasNext(); ) {
                    Map.Entry e = (Map.Entry)i.next();
                    SetOfLocations this_s = (SetOfLocations)this.outside_edges.get(e.getKey());
                    SetOfLocations that_s = (SetOfLocations)e.getValue();
                    if (this_s == null) {
                        if (TRACE_UNION) out_ta.println(this+" doesn't contain outside edge on "+e.getKey()+", adding");
                        this_s = that_s.copy_deep(old_to_new);
                        for (Iterator j=this_s.iterator(); j.hasNext(); ) {
                            ((ProgramLocation)j.next()).methodSequences.setExposed();
                        }
                        this.outside_edges.put(e.getKey(), this_s);
                        change = true;
                    } else {
                        if (TRACE_UNION) out_ta.println(this+" already contains outside edge on "+e.getKey()+", unioning");
                        if (this_s.union_deep(that_s, old_to_new, stack)) change = true;
                    }
                }
            }
            return change;
        }
        public final boolean union_deep(ProgramLocation that, HashMap old_to_new, Stack stack) {
            if (this == that) return false;
            boolean b = super.union_deep(that, old_to_new, stack);
            if (stack.contains(this)) {
                if (TRACE_UNION) out_ta.println("Cycle in outside location "+this+"!");
                return false;
            }
            stack.push(this);
            if (unionOutsideEdges_deep((OutsideProgramLocation)that, old_to_new, stack)) b = true;
            Object o = stack.pop();
            Assert._assert(o == this);
            return b;
        }
        public final boolean isFromOutside() { return true; }
    }
    
    static class AllocationLocation extends ProgramLocation {
        jq_Type createdType; jq_Method method; int bcIndex;
        AllocationLocation(jq_Type t, jq_Method m, int bc) {
            super();
            createdType = t; method = m; bcIndex = bc;
        }
        public ProgramLocation copy_deep(HashMap old_to_new) {
            AllocationLocation that = (AllocationLocation)old_to_new.get(IdentityHashCodeWrapper.create(this));
            if (that == null) {
                if (TRACE_COPY) out_ta.println("New location to copy deep: "+this);
                that = new AllocationLocation(this.createdType, this.method, this.bcIndex);
                that.methodSequences = this.methodSequences.copy_deep();
                old_to_new.put(IdentityHashCodeWrapper.create(this), that);
                this.copyInsideEdges_deep(that, old_to_new);
            }
            return that;
        }
        public ProgramLocation copy_shallow(HashMap old_to_new) {
            AllocationLocation that = (AllocationLocation)old_to_new.get(IdentityHashCodeWrapper.create(this));
            if (that == null) {
                if (TRACE_COPY) out_ta.println("New location to copy shallow: "+this);
                that = new AllocationLocation(this.createdType, this.method, this.bcIndex);
                //that.methodSequences = this.methodSequences.copy_deep();
                that.methodSequences = new MethodCallSequences(true);
                old_to_new.put(IdentityHashCodeWrapper.create(this), that);
            }
            return that;
        }
        public jq_Type getType() { return createdType; }
        public ProgramLocation filterByType(jq_Type t) {
            if (t == null) return this;
            createdType.prepare();
            t.prepare();
            if (TypeCheck.isAssignable(createdType, t)) return this;
            if (TRACE_FILTER) out_ta.println("Filtered out "+this+" because it is not a subtype of "+t);
            return null;
        }
        public boolean equals(AllocationLocation that) {
            return (method == that.method) && (bcIndex == that.bcIndex);
        }
        public boolean equals(Object o) {
            if (o instanceof AllocationLocation) return equals((AllocationLocation)o);
            return false;
        }
        public int hashCode() { return method.hashCode() ^ bcIndex; }
        public String toString() { return "ocs:"+createdType+"@"+method.getName()+":"+bcIndex+" "+methodSequences; }
    }
    static class LoadConstantLocation extends ProgramLocation {
        Object constant; jq_Type type; jq_Method method; int bcIndex;
        LoadConstantLocation(Object o, jq_Type t, jq_Method m, int bc) {
            super();
            constant = o; type = t; method = m; bcIndex = bc;
        }
        public ProgramLocation copy_deep(HashMap old_to_new) {
            LoadConstantLocation that = (LoadConstantLocation)old_to_new.get(IdentityHashCodeWrapper.create(this));
            if (that == null) {
                if (TRACE_COPY) out_ta.println("New location to copy deep: "+this);
                that = new LoadConstantLocation(this.constant, this.type, this.method, this.bcIndex);
                that.methodSequences = this.methodSequences.copy_deep();
                // no edges from constant object.
                old_to_new.put(IdentityHashCodeWrapper.create(this), that);
            }
            return that;
        }
        public ProgramLocation copy_shallow(HashMap old_to_new) {
            LoadConstantLocation that = (LoadConstantLocation)old_to_new.get(IdentityHashCodeWrapper.create(this));
            if (that == null) {
                if (TRACE_COPY) out_ta.println("New location to copy shallow: "+this);
                that = new LoadConstantLocation(this.constant, this.type, this.method, this.bcIndex);
                //that.methodSequences = this.methodSequences.copy_deep();
                that.methodSequences = new MethodCallSequences(true);
                old_to_new.put(IdentityHashCodeWrapper.create(this), that);
            }
            return that;
        }
        public jq_Type getType() { return type; }
        public ProgramLocation filterByType(jq_Type t) {
            if ((type == null) || (t == null)) return this;
            type.prepare();
            t.prepare();
            if (TypeCheck.isAssignable(type, t)) return this;
            if (TRACE_FILTER) out_ta.println("Filtered out "+this+" because it is not a subtype of "+t);
            return null;
        }
        public boolean equals(LoadConstantLocation that) {
            return this.constant == that.constant;
        }
        public boolean equals(Object o) {
            if (o instanceof LoadConstantLocation) return equals((LoadConstantLocation)o);
            return false;
        }
        public int hashCode() { return constant==null?0:constant.hashCode(); }
        public String toString() { return "lc:"+constant+" "+methodSequences; }
    }
    static class ParamLocation extends OutsideProgramLocation {
        jq_Method method; int paramNum; jq_Type declaredType;
        ParamLocation(jq_Method m, int n, jq_Type type) {
            method = m; paramNum = n; declaredType = type;
        }
        public jq_Type getOriginalType() { return method.getParamTypes()[paramNum]; }
        public ProgramLocation copy_deep(HashMap old_to_new) {
            ParamLocation that = (ParamLocation)old_to_new.get(IdentityHashCodeWrapper.create(this));
            if (that == null) {
                if (TRACE_COPY) out_ta.println("New location to copy deep: "+this);
                that = new ParamLocation(this.method, this.paramNum, this.declaredType);
                that.methodSequences = this.methodSequences.copy_deep();
                old_to_new.put(IdentityHashCodeWrapper.create(this), that);
                this.copyInsideEdges_deep(that, old_to_new);
                this.copyOutsideEdges_deep(that, old_to_new);
            }
            return that;
        }
        public ProgramLocation copy_shallow(HashMap old_to_new) {
            ParamLocation that = (ParamLocation)old_to_new.get(IdentityHashCodeWrapper.create(this));
            if (that == null) {
                if (TRACE_COPY) out_ta.println("New location to copy shallow: "+this);
                that = new ParamLocation(this.method, this.paramNum, this.declaredType);
                //that.methodSequences = this.methodSequences.copy_deep();
                that.methodSequences = new MethodCallSequences(true);
                old_to_new.put(IdentityHashCodeWrapper.create(this), that);
            }
            return that;
        }
        public jq_Type getType() { return declaredType; }
        public ProgramLocation filterByType(jq_Type t) {
            if (t == null) return this;
            declaredType.prepare();
            t.prepare();
            if (TypeCheck.isAssignable(declaredType, t)) return this;
            ParamLocation that = new ParamLocation(this.method, this.paramNum, t);
            that.methodSequences = this.methodSequences.copy_deep();
            if (TRACE_FILTER) out_ta.println("Created subtype of parameter with type "+t+": "+that);
            return that;
        }
        public boolean equals(ParamLocation that) {
            return (method == that.method) && (paramNum == that.paramNum);
        }
        public boolean equals(Object o) {
            if (o instanceof ParamLocation) return equals((ParamLocation)o);
            return false;
        }
        public int hashCode() { return method.hashCode() ^ paramNum; }
        public String toString() { return method.getName()+":p"+paramNum+" "+methodSequences;  }
    }
    static class StaticFieldLocation extends OutsideProgramLocation {
        jq_StaticField field; jq_Type type;
        StaticFieldLocation(jq_StaticField f, jq_Type t) {
            super();
            field = f; type = t;
        }
        public jq_Type getOriginalType() { return field.getType(); }
        public ProgramLocation copy_deep(HashMap old_to_new) {
            StaticFieldLocation that = (StaticFieldLocation)old_to_new.get(IdentityHashCodeWrapper.create(this));
            if (that == null) {
                if (TRACE_COPY) out_ta.println("New location to copy deep: "+this);
                that = new StaticFieldLocation(this.field, this.type);
                that.methodSequences = this.methodSequences.copy_deep();
                old_to_new.put(IdentityHashCodeWrapper.create(this), that);
                this.copyInsideEdges_deep(that, old_to_new);
                this.copyOutsideEdges_deep(that, old_to_new);
            }
            return that;
        }
        public ProgramLocation copy_shallow(HashMap old_to_new) {
            StaticFieldLocation that = (StaticFieldLocation)old_to_new.get(IdentityHashCodeWrapper.create(this));
            if (that == null) {
                if (TRACE_COPY) out_ta.println("New location to copy shallow: "+this);
                that = new StaticFieldLocation(this.field, this.type);
                //that.methodSequences = this.methodSequences.copy_deep();
                that.methodSequences = new MethodCallSequences(true);
                old_to_new.put(IdentityHashCodeWrapper.create(this), that);
            }
            return that;
        }
        public jq_Type getType() { return type; }
        public ProgramLocation filterByType(jq_Type t) {
            if (t == null) return this;
            type.prepare();
            t.prepare();
            if (TypeCheck.isAssignable(type, t)) return this;
            StaticFieldLocation that = new StaticFieldLocation(this.field, t);
            that.methodSequences = this.methodSequences.copy_deep();
            if (TRACE_FILTER) out_ta.println("Created subtype of static field with type "+t+": "+that);
            return that;
        }
        public boolean equals(StaticFieldLocation that) {
            return (field == that.field);
        }
        public boolean equals(Object o) {
            if (o instanceof StaticFieldLocation) return equals((StaticFieldLocation)o);
            return false;
        }
        public int hashCode() { return field.hashCode(); }
        public String toString() { return "sf:"+field.getName()+" "+methodSequences;  }
    }
    static class DereferenceLocation extends OutsideProgramLocation {
        Dereference deref; jq_Method method; int bcIndex; jq_Type type;
        DereferenceLocation(Dereference d, jq_Method m, int bc, jq_Type t) {
            super();
            deref = d; method = m; bcIndex = bc; type = t;
        }
        public jq_Type getOriginalType() { return deref.getType(); }
        public ProgramLocation copy_deep(HashMap old_to_new) {
            DereferenceLocation that = (DereferenceLocation)old_to_new.get(IdentityHashCodeWrapper.create(this));
            if (that == null) {
                if (TRACE_COPY) out_ta.println("New location to copy deep: "+this);
                that = new DereferenceLocation(this.deref, this.method, this.bcIndex, this.type);
                that.methodSequences = this.methodSequences.copy_deep();
                old_to_new.put(IdentityHashCodeWrapper.create(this), that);
                this.copyInsideEdges_deep(that, old_to_new);
                this.copyOutsideEdges_deep(that, old_to_new);
            }
            return that;
        }
        public ProgramLocation copy_shallow(HashMap old_to_new) {
            DereferenceLocation that = (DereferenceLocation)old_to_new.get(IdentityHashCodeWrapper.create(this));
            if (that == null) {
                if (TRACE_COPY) out_ta.println("New location to copy shallow: "+this);
                that = new DereferenceLocation(this.deref, this.method, this.bcIndex, this.type);
                //that.methodSequences = this.methodSequences.copy_deep();
                that.methodSequences = new MethodCallSequences(true);
                old_to_new.put(IdentityHashCodeWrapper.create(this), that);
            }
            return that;
        }
        public jq_Type getType() { return type; }
        public ProgramLocation filterByType(jq_Type t) {
            if (t == null) return this;
            type.prepare();
            t.prepare();
            if (TypeCheck.isAssignable(type, t)) return this;
            DereferenceLocation that = new DereferenceLocation(this.deref, this.method, this.bcIndex, t);
            that.methodSequences = this.methodSequences.copy_deep();
            if (TRACE_FILTER) out_ta.println("Created subtype of deref with type "+t+": "+that);
            return that;
        }
        public boolean equals(DereferenceLocation that) {
            return (method == that.method) &&
                   ((bcIndex == -1) || (that.bcIndex == -1) || (bcIndex == that.bcIndex)) &&
                   //(type == that.type) &&
                   deref.equals(that.deref);
        }
        public boolean equals(Object o) {
            if (o instanceof DereferenceLocation) return equals((DereferenceLocation)o);
            return false;
        }
        public int hashCode() { return deref.hashCode() ^ method.hashCode(); }
        //public String toString() { return "dl:"+deref+"("+method.getName()+":"+bcIndex+")"+" "+methodSequences+"@"+Integer.toHexString(System.identityHashCode(this)); }
        public String toString() { return "dl:"+deref+"(type="+type+")@"+method.getName()+":"+bcIndex+" "+methodSequences; }
    }
    /*
    static class ReturnLocation extends ProgramLocation {
        jq_Method method;
        public boolean equals(ReturnLocation that) {
            return (method == that.method);
        }
        public boolean equals(Object o) {
            if (o instanceof ReturnLocation) return equals((ReturnLocation)o);
            return false;
        }
        public int hashCode() { return method.hashCode(); }
        public String toString() { return method.getName()+"(returned)"; }
    }
    static class ThrownLocation extends ProgramLocation {
        jq_Method method;
        public boolean equals(ThrownLocation that) {
            return (method == that.method);
        }
        public boolean equals(Object o) {
            if (o instanceof ThrownLocation) return equals((ThrownLocation)o);
            return false;
        }
        public int hashCode() { return method.hashCode(); }
        public String toString() { return method.getName()+"(thrown)"; }
    }
    */
    static class CaughtLocation extends OutsideProgramLocation {
        jq_Method method; int bcIndex; jq_Type type, originalType;
        CaughtLocation(jq_Method m, int bc, jq_Type t) {
            super();
            method = m; bcIndex = bc; originalType = type = t;
        }
        public jq_Type getOriginalType() { return originalType; }
        public ProgramLocation copy_deep(HashMap old_to_new) {
            CaughtLocation that = (CaughtLocation)old_to_new.get(IdentityHashCodeWrapper.create(this));
            if (that == null) {
                if (TRACE_COPY) out_ta.println("New location to copy deep: "+this);
                that = new CaughtLocation(this.method, this.bcIndex, this.type);
                that.methodSequences = this.methodSequences.copy_deep();
                old_to_new.put(IdentityHashCodeWrapper.create(this), that);
                this.copyInsideEdges_deep(that, old_to_new);
                this.copyOutsideEdges_deep(that, old_to_new);
            }
            return that;
        }
        public ProgramLocation copy_shallow(HashMap old_to_new) {
            CaughtLocation that = (CaughtLocation)old_to_new.get(IdentityHashCodeWrapper.create(this));
            if (that == null) {
                if (TRACE_COPY) out_ta.println("New location to copy shallow: "+this);
                that = new CaughtLocation(this.method, this.bcIndex, this.type);
                //that.methodSequences = this.methodSequences.copy_deep();
                that.methodSequences = new MethodCallSequences(true);
                old_to_new.put(IdentityHashCodeWrapper.create(this), that);
            }
            return that;
        }
        public jq_Type getType() { return type; }
        public ProgramLocation filterByType(jq_Type t) {
            if (t == null) return this;
            type.prepare();
            t.prepare();
            if (TypeCheck.isAssignable(type, t)) return this;
            CaughtLocation that = new CaughtLocation(this.method, this.bcIndex, t);
            that.methodSequences = this.methodSequences.copy_deep();
            if (TRACE_FILTER) out_ta.println("Created subtype of caught exception with type "+t+": "+that);
            return that;
        }
        public boolean equals(CaughtLocation that) {
            return (method == that.method) && (bcIndex == that.bcIndex);
        }
        public boolean equals(Object o) {
            if (o instanceof CaughtLocation) return equals((CaughtLocation)o);
            return false;
        }
        public int hashCode() { return method.hashCode() ^ bcIndex; }
        public String toString() { return method.getName()+":"+bcIndex+"(caught exception)"; }
    }
    
    static class MethodCall {
        
        jq_Method caller; int bc_i;
        jq_Method method;
        
        MethodCall(jq_Method caller, int bc_i, jq_Method method) {
            this.caller = caller; this.bc_i = bc_i; this.method = method;
        }
        
        jq_Method getMethod() { return method; }
        public Utf8 getName() { return method.getName(); }
        public String toString() {
            String className = method.getDeclaringClass().getJDKName();
            className = className.substring(className.lastIndexOf('.')+1);
            return className+"."+method.getName()+"()@"+caller.getName()+":"+bc_i;
        }
        public boolean equals(MethodCall that) {
            return this.method == that.method;
        }
        public boolean equals(Object that) {
            return this.equals((MethodCall)that);
        }
        public int hashCode() { return method.hashCode(); }
        public MethodCall asType(jq_Class k) {
            jq_Method m = (jq_Method)k.getDeclaredMember(method.getNameAndDesc());
            return new MethodCall(this.caller, this.bc_i, m);
        }
    }
    
    public static class MethodCallSequences {
        private boolean exposed;
        private Map/*<jq_Class, MethodCallSequence>*/ modelsByType;
        
        MethodCallSequences(boolean outside) {
            this.exposed = outside;
            this.modelsByType = new HashMap();
        }

        boolean isEmpty() {
            return modelsByType.size() == 0;
        }
        
        void setExposed() {
            exposed = true;
            for (Iterator i=this.modelsByType.entrySet().iterator(); i.hasNext(); ) {
                Map.Entry e = (Map.Entry)i.next();
                MethodCallSequence mcs = (MethodCallSequence)e.getValue();
                mcs.setExposed();
            }
        }
        
        MethodCallSequences copy_deep() {
            MethodCallSequences that = new MethodCallSequences(this.exposed);
            for (Iterator i=this.modelsByType.entrySet().iterator(); i.hasNext(); ) {
                Map.Entry e = (Map.Entry)i.next();
                MethodCallSequence mcs = (MethodCallSequence)e.getValue();
                that.modelsByType.put(e.getKey(), mcs.copy_deep());
            }
            return that;
        }
        
        private void addAllSubClasses(jq_Class k, jq_Method m, LinearSet s) {
            jq_Class[] subs = k.getSubClasses();
            for (int i=0; i<subs.length; ++i) {
                jq_Class sub = subs[i];
                if (sub.getDeclaredMember(m.getNameAndDesc()) != null) s.add(sub);
                addAllSubClasses(sub, m, s);
            }
        }
        
        private void addAllSubInterfaces(jq_Class k, jq_Method m, LinearSet s) {
            jq_Class[] subs = k.getSubInterfaces();
            for (int i=0; i<subs.length; ++i) {
                jq_Class sub = subs[i];
                if (sub.getDeclaredMember(m.getNameAndDesc()) != null) s.add(sub);
                addAllSubInterfaces(sub, m, s);
            }
        }
        
        private void addAllSuperInterfaces(jq_Class k, jq_Method m, LinearSet s) {
            k.prepare();
            jq_Class[] subs = k.getInterfaces();
            for (int i=0; i<subs.length; ++i) {
                jq_Class sub = subs[i];
                if (sub.getDeclaredMember(m.getNameAndDesc()) != null) s.add(sub);
                addAllSuperInterfaces(sub, m, s);
            }
        }
        
        Iterator getEffectiveTypes(jq_Type t, jq_Method m) {
            LinearSet types = new LinearSet();
            if (t.isClassType()) {
                // self and all superclasses
                for (jq_Class k = (jq_Class)t; k != null; k = k.getSuperclass()) {
                    k.load();
                    if (k.getDeclaredMember(m.getNameAndDesc()) != null) types.add(k);
                }
                // all subclasses
                addAllSubClasses((jq_Class)t, m, types);
                if (((jq_Class)t).isInterface()) {
                    // all superinterfaces
                    addAllSuperInterfaces((jq_Class)t, m, types);
                    // all subinterfaces
                    addAllSubInterfaces((jq_Class)t, m, types);
                }
            }
            return types.iterator();
        }
        
        void mustCall(String loc, jq_Type t, MethodCall m) {
            for (Iterator i=getEffectiveTypes(t, m.getMethod()); i.hasNext(); ) {
                jq_Class k = (jq_Class)i.next();
                MethodCallSequence mcs = (MethodCallSequence)modelsByType.get(k);
                if (mcs == null) {
                    modelsByType.put(k, mcs = new MethodCallSequence(exposed));
                }
                MethodCall m2 = m.asType(k);
                mcs.mustCall(loc, k, m2);
            }
        }
        void mayCall(String loc, jq_Type t, MethodCall m) {
            for (Iterator i=getEffectiveTypes(t, m.getMethod()); i.hasNext(); ) {
                jq_Class k = (jq_Class)i.next();
                MethodCallSequence mcs = (MethodCallSequence)modelsByType.get(k);
                if (mcs == null) {
                    modelsByType.put(k, mcs = new MethodCallSequence(exposed));
                }
                MethodCall m2 = m.asType(k);
                mcs.mayCall(loc, k, m2);
            }
        }
        
        void addFirstCalledToMayCall(String loc, jq_Type t, MethodCallSequences that) {
            for (Iterator i=that.modelsByType.entrySet().iterator(); i.hasNext(); ) {
                Map.Entry e = (Map.Entry)i.next();
                jq_Class k = (jq_Class)e.getKey();
                MethodCallSequence that_mcs = (MethodCallSequence)e.getValue();
                MethodCallSequence this_mcs = (MethodCallSequence)this.modelsByType.get(k);
                if (this_mcs == null) {
                    modelsByType.put(k, this_mcs = new MethodCallSequence(exposed));
                }
                //out_ta.println("Adding "+k+" set "+that_mcs.firstCalled+" to may call for "+this_mcs);
                this_mcs.addToMayCall(loc, k, that_mcs.firstCalled);
            }
        }
        
        void updateMustCall(jq_Type t, MethodCallSequences that) {
            for (Iterator i=that.modelsByType.entrySet().iterator(); i.hasNext(); ) {
                Map.Entry e = (Map.Entry)i.next();
                jq_Class k = (jq_Class)e.getKey();
                MethodCallSequence that_mcs = (MethodCallSequence)e.getValue();
                MethodCallSequence this_mcs = (MethodCallSequence)this.modelsByType.get(k);
                if (this_mcs == null) {
                    modelsByType.put(k, this_mcs = new MethodCallSequence(exposed));
                }
                //out_ta.println("Updating "+k+" set "+this_mcs+" with must call from "+that_mcs);
                this_mcs.updateMustCall(that_mcs);
                //out_ta.println("Result: "+this_mcs);
            }
        }
        
        boolean union_deep(MethodCallSequences that) {
            boolean change = false;
            if (that.exposed && !this.exposed) {
                change = true;
                this.exposed = true;
            }
            for (Iterator i=that.modelsByType.entrySet().iterator(); i.hasNext(); ) {
                Map.Entry e = (Map.Entry)i.next();
                MethodCallSequence that_mcs = (MethodCallSequence)e.getValue();
                MethodCallSequence this_mcs = (MethodCallSequence)this.modelsByType.get(e.getKey());
                if (this_mcs == null) {
                    this.modelsByType.put(e.getKey(), this_mcs = that_mcs.copy_deep());
                    change = true;
                } else {
                    if (this_mcs.union_deep(that_mcs)) change = true;
                }
            }
            return change;
        }
        
        boolean intersect_deep(MethodCallSequences that) {
            boolean change = false;
            if (!that.exposed && this.exposed) {
                change = true;
                this.exposed = false;
            }
            for (Iterator i=that.modelsByType.entrySet().iterator(); i.hasNext(); ) {
                Map.Entry e = (Map.Entry)i.next();
                MethodCallSequence that_mcs = (MethodCallSequence)e.getValue();
                MethodCallSequence this_mcs = (MethodCallSequence)this.modelsByType.get(e.getKey());
                if (this_mcs == null) {
                    this.modelsByType.put(e.getKey(), this_mcs = that_mcs.copy_deep());
                    change = true;
                } else {
                    if (this_mcs.intersect_deep(that_mcs)) change = true;
                }
            }
            return change;
        }
        
        public String toString() {
            StringBuffer sb = new StringBuffer();
            sb.append("{");
            if (exposed) sb.append("outer+");
            for (Iterator i=this.modelsByType.entrySet().iterator(); i.hasNext(); ) {
                Map.Entry e = (Map.Entry)i.next();
                sb.append(" TypeModel:"+e.getKey());
                sb.append(" Last:"+e.getValue());
            }
            sb.append("}");
            return sb.toString();
        }
        public String printFirstCalled() {
            StringBuffer sb = new StringBuffer();
            sb.append("{");
            for (Iterator i=this.modelsByType.entrySet().iterator(); i.hasNext(); ) {
                Map.Entry e = (Map.Entry)i.next();
                sb.append(" TypeModel:"+e.getKey());
                sb.append(" First:"+((MethodCallSequence)e.getValue()).getFirstCalled());
            }
            sb.append("}");
            return sb.toString();
        }
    }
        
    public static class MethodCallSequence {
        private boolean exposed;
        private Set/*<MethodCall>*/ firstCalled;
        private Set/*<MethodCall>*/ lastCalled;
        
        MethodCallSequence(boolean outside) {
            this.firstCalled = new LinearSet();
            this.lastCalled = new LinearSet();
            this.exposed = outside;
        }
        MethodCallSequence(MethodCallSequence that) {
            this.firstCalled = new LinearSet(that.firstCalled);
            this.lastCalled = new LinearSet(that.lastCalled);
            this.exposed = that.exposed;
        }
        
        void setExposed() { exposed = true; }
        
        boolean union_deep(MethodCallSequence that) {
            boolean change = false;
            if (that.exposed && !this.exposed) {
                change = true;
                this.exposed = true;
            }
            if (this.firstCalled.addAll(that.firstCalled)) change = true;
            if (this.lastCalled.addAll(that.lastCalled)) change = true;
            return change;
        }
        
        boolean intersect_deep(MethodCallSequence that) {
            boolean change = false;
            if (!that.exposed && this.exposed) {
                change = true;
                this.exposed = false;
            }
            if (this.firstCalled.addAll(that.firstCalled)) change = true;
            if (this.lastCalled.addAll(that.lastCalled)) change = true;
            return change;
        }

        /*
        void mustCall(String loc, jq_Type t, Set s) {
            if (exposed) { firstCalled.addAll(s); }
            exposed = false;
            if (modeler != null) modeler.checkModel(loc, t, lastCalled, s);
            lastCalled.clear();
            lastCalled.addAll(s);
        }
         */
        void mustCall(String loc, jq_Type t, MethodCall m) {
            if (exposed) { firstCalled.add(m); }
            exposed = false;
            if (modeler != null) modeler.checkModel(loc, t, lastCalled, m);
            lastCalled.clear();
            lastCalled.add(m);
        }
        
        /*
        void mayCall(String loc, jq_Type t, Set s) {
            if (exposed) { firstCalled.addAll(s); }
            if (modeler != null) modeler.checkModel(loc, t, lastCalled, s);
            lastCalled.addAll(s);
        }
         */
        void mayCall(String loc, jq_Type t, MethodCall m) {
            if (exposed) { firstCalled.add(m); }
            if (modeler != null) modeler.checkModel(loc, t, lastCalled, m);
            lastCalled.add(m);
        }

        void updateMustCall(MethodCallSequence that) {
            if (this.exposed) { this.firstCalled.addAll(that.lastCalled); }
            if (!that.exposed) {
                this.exposed = false; this.lastCalled.clear();
            }
            this.lastCalled.addAll(that.lastCalled);
        }
        
        void addToMayCall(String loc, jq_Type t, Set m) {
            if (exposed) { firstCalled.addAll(m); }
            if (modeler != null) modeler.checkModel(loc, t, lastCalled, m);
        }
        
        public String toString() {
            StringBuffer s = new StringBuffer("{");
            for (Iterator i=lastCalled.iterator(); i.hasNext(); ) {
                MethodCall m = (MethodCall)i.next();
                s.append(m.getName().toString());
                if (i.hasNext()) s.append(',');
            }
            if (exposed) s.append("+outer");
            s.append('}');
            //s.append(Integer.toHexString(this.hashCode()));
            return s.toString();
        }
        
        MethodCallSequence copy_deep() {
            return new MethodCallSequence(this);
        }
        
        boolean isExposed() { return exposed; }
        
        Set getFirstCalled() { return firstCalled; }
        Set getLastCalled() { return lastCalled; }
        
        public static Modeler modeler;
        
        public abstract static class Modeler {
            public void checkModel(String loc, jq_Type type, Set lastCalled, Set s) {
                for (Iterator i=s.iterator(); i.hasNext(); ) {
                    checkModel(loc, type, lastCalled, (MethodCall)i.next());
                }
            }
            public abstract void checkModel(String loc, jq_Type type, Set lastCalled, MethodCall m);
            public abstract void dump() throws java.io.IOException;
        }
        
        /*** TEMPORARILY COMMENT OUT UNTIL WE CHECK IN DETOX STUFF. ***
        public static class DetoxModelBuilder extends Modeler {
            
            public static boolean TRACE_BUILDMODEL = false;
            
            public detox.model.ProgramModel model;
            
            public DetoxModelBuilder() {
                model = new detox.model.ProgramModel();
            }
            
            static String methodToString(jq_Method m) {
                return m.getName().toString();
            }
            
            public void checkModel(String loc, jq_Type type, Set lastCalled, MethodCall m) {
                if (!(type instanceof jq_Class)) return;
                jq_Class klass = (jq_Class)type;
                String klassName = klass.getJDKName();
                detox.model.Graph g = (detox.model.Graph)model.types.get(klassName);
                if (g == null) {
                    model.addGraph(g = new detox.model.Graph(klassName));
                }
                jq_Method after_jq_m = m.getMethod();
                String after = methodToString(after_jq_m);
                for (Iterator i = lastCalled.iterator(); i.hasNext(); ) {
                    MethodCall before_m = (MethodCall)i.next();
                    jq_Method before_jq_m = before_m.getMethod();
                    String before = methodToString(before_jq_m);
                    if (TRACE_BUILDMODEL) {
                        if (!g.hasEdge(before, after)) {
                            out_ta.println("NEW EDGE: model "+klassName+" "+before+"->"+after+", location: "+loc);
                        }
                    }
                    g.bumpEdge(before, after);
                }
            }
            
            public void dump() throws java.io.IOException {
                String fileBase = System.getProperty("model", "model.xml");
                java.io.File f;
                for(int i=0; ; ) {
                    f = new java.io.File(fileBase + (i == 0 ? "" : String.valueOf(i)));
                    i++;
                    if (f.createNewFile()) break;
                }
                java.io.PrintStream ps = new java.io.PrintStream(new java.io.FileOutputStream(f));
                // Produce XML File
                ps.print(model.toXML());
                ps.close();
            }
        }
        
        public static class DetoxModelChecker extends Modeler {
            
            public detox.model.LegalityModel model;
            public static boolean PRINT_MODEL = false;
            public static boolean DUMP_ALL = false;
            
            public DetoxModelChecker(detox.model.LegalityModel model) {
                this.model = model;
                printModel();
            }
            
            public DetoxModelChecker() {
                this(getModel());
            }
            
            public static detox.model.LegalityModel getModel() {
                detox.model.ProgramModel program_model = new detox.model.LoadingModel();
                String filename = System.getProperty("model", "model.log");
                program_model.incorporateFile(filename);
                return new detox.model.StrictLegalityModel(program_model);
            }
            
            void printModel() {
                if (PRINT_MODEL) {
                    System.err.println(model.modelDoc());
                }
            }

            static String methodToString(jq_Method m) {
                return m.getName().toString();
            }
            
            public void checkModel(String loc, jq_Type type, Set lastCalled, MethodCall m) {
                if (!(type instanceof jq_Class)) return;
                jq_Class klass = (jq_Class)type;
                String klassName = klass.getJDKName();
                jq_Method after_jq_m = m.getMethod();
                String after = methodToString(after_jq_m);
                boolean isTargetKnown = model.hasNode(klassName, after);
                for (Iterator i = lastCalled.iterator(); i.hasNext(); ) {
                    MethodCall before_m = (MethodCall)i.next();
                    jq_Method before_jq_m = before_m.getMethod();
                    String before = methodToString(before_jq_m);
                    boolean isSourceKnown = model.hasNode(klassName, before);
                    Object pair = Default.pair(before_jq_m, after_jq_m);
                    boolean already_reported = reported.contains(pair);
                    if (!already_reported) reported.add(pair);
                    if (!isSourceKnown) {
                        if (!isTargetKnown) {
                            ++nUnknownToUnknown;
                            if (!already_reported) {
                                ++nUniqueUnknownToUnknown;
                            }
                        } else {
                            ++nUnknownToKnown;
                            if (!already_reported) {
                                out_ta.println("New UNKNOWN->KNOWN: at "+loc+" on class: "+klass+" ==> "+before+"->"+after);
                                ++nUniqueUnknownToKnown;
                            } else if (DUMP_ALL) {
                                out_ta.println("UNKNOWN->KNOWN: at "+loc+" on class: "+klass+" ==> "+before+"->"+after);
                            }
                        }
                        continue;
                    }
                    if (!isTargetKnown) {
                        ++nKnownToUnknown;
                        if (!already_reported) {
                            out_ta.println("New KNOWN->UNKNOWN: at "+loc+" on class: "+klass+" ==> "+before+"->"+after);
                            ++nUniqueKnownToUnknown;
                        } else if (DUMP_ALL) {
                            out_ta.println("KNOWN->UNKNOWN: at "+loc+" on class: "+klass+" ==> "+before+"->"+after);
                        }
                        continue;
                    }
                    if (!model.legalCall(klassName, before, after)) {
                        ++nKnownToKnown_illegal;
                        if (!already_reported) {
                            out_ta.println("New ILLEGAL TRANSITION: at "+loc+" on class: "+klass+" ==> "+before+"->"+after);
                            ++nUniqueKnownToKnown_illegal;
                        } else if (DUMP_ALL) {
                            out_ta.println("ILLEGAL TRANSITION: at "+loc+" on class: "+klass+" ==> "+before+"->"+after);
                        }
                    } else {
                        ++nKnownToKnown_legal;
                        if (!already_reported) {
                            ++nUniqueKnownToKnown_legal;
                        }
                    }
                }
            }

            public static HashSet reported = new HashSet();
            
            public static int nUnknownToUnknown;
            public static int nUnknownToKnown;
            public static int nKnownToUnknown;
            public static int nKnownToKnown_legal;
            public static int nKnownToKnown_illegal;
            
            public static int nUniqueUnknownToUnknown;
            public static int nUniqueUnknownToKnown;
            public static int nUniqueKnownToUnknown;
            public static int nUniqueKnownToKnown_legal;
            public static int nUniqueKnownToKnown_illegal;
            
            public void dump() {
                out_ta.println("unknown->unknown transitions:\t"+nUnknownToUnknown+" ("+nUniqueUnknownToUnknown+" unique)");
                out_ta.println("unknown->known transitions:\t"+nUnknownToKnown+" ("+nUniqueUnknownToKnown+" unique)");
                out_ta.println("known->unknown transitions:\t"+nKnownToUnknown+" ("+nUniqueKnownToUnknown+" unique)");
                out_ta.println("known->known legal transitions: "+nKnownToKnown_legal+" ("+nUniqueKnownToKnown_legal+" unique)");
                out_ta.println("known->known illegal transitions: "+nKnownToKnown_illegal+" ("+nUniqueKnownToKnown_illegal+" unique)");
            }

        }
         *** TEMPORARILY COMMENT OUT UNTIL WE CHECK IN DETOX STUFF. ***/

    }
    
    static class AnalysisState {
        ParamLocation[] parameters;
        Map/*<jq_StaticField, SetOfLocations>*/ static_variables;
        SetOfLocations[] variables;
        
        public static final boolean TRACE_VARIABLES = false;
        public static final boolean TRACE_STATIC = false;
        
        static AnalysisState makeEntry(jq_Method m) {
            AnalysisState dis = new AnalysisState();
            jq_Type[] paramTypes = m.getParamTypes();
            int nLocals = m.getMaxLocals();
            int nStack = m.getMaxStack();
            dis.parameters = new ParamLocation[paramTypes.length];
            dis.variables = new SetOfLocations[nLocals+nStack];
            for (int i=0, j=0; i<paramTypes.length; ++i, ++j) {
                jq_Type p = paramTypes[i];
                if (p.isReferenceType()) {
                    ParamLocation pl = new ParamLocation(m, i, p);
                    pl.methodSequences = new MethodCallSequences(true);
                    dis.parameters[i] = pl;
                    dis.variables[j] = SetOfLocations.makeParamSet(pl);
                }
                else if (p.getReferenceSize() == 8) ++j;
            }
            dis.static_variables = new HashMap();
            return dis;
        }
        static AnalysisState makeExit(jq_Method m) {
            AnalysisState dis = new AnalysisState();
            jq_Type[] paramTypes = m.getParamTypes();
            dis.parameters = new ParamLocation[paramTypes.length];
            for (int i=0, j=0; i<paramTypes.length; ++i, ++j) {
                jq_Type p = paramTypes[i];
                if (p.isReferenceType()) {
                    ParamLocation pl = new ParamLocation(m, i, p);
                    pl.methodSequences = new MethodCallSequences(true);
                    dis.parameters[i] = pl;
                }
                else if (p.getReferenceSize() == 8) ++j;
            }
            dis.static_variables = new HashMap();
            return dis;
        }
        private AnalysisState() {}
        
        AnalysisState(AnalysisState that) {
            this.parameters = new ParamLocation[that.parameters.length];
            HashMap old_to_new = new HashMap();
            for (int i=0; i<this.parameters.length; ++i) {
                if (that.parameters[i] != null)
                    this.parameters[i] = (ParamLocation)that.parameters[i].copy_deep(old_to_new);
            }
            if (that.variables != null) {
                this.variables = new SetOfLocations[that.variables.length];
                for (int i=0; i<this.variables.length; ++i) {
                    if (that.variables[i] != null)
                        this.variables[i] = that.variables[i].copy_deep(old_to_new);
                }
            }
            this.static_variables = new HashMap();
            for (Iterator i=that.static_variables.entrySet().iterator(); i.hasNext(); ) {
                Map.Entry e = (Map.Entry)i.next();
                SetOfLocations t = (SetOfLocations)e.getValue();
                this.static_variables.put(e.getKey(), t.copy_deep(old_to_new));
            }
        }
        
        public SetOfLocations get(int i) {
            if (TRACE_VARIABLES) out_ta.println("getting "+i+": "+variables[i]);
            return variables[i];
        }
        
        public void put(int i, SetOfLocations t) {
            if (TRACE_VARIABLES) out_ta.println("putting "+i+": "+t+" old value: "+variables[i]);
            if (t != null) t = t.copy_shallow();
            variables[i] = t;
        }
        
        public void move(int to, int from) {
            if (TRACE_VARIABLES) out_ta.println("moving "+from+" to "+to+": "+variables[from]+" to "+variables[to]);
            // can be null when storing jsr retaddr
            //variables[to] = (variables[from]==null)?null:variables[from].copy_shallow();
            variables[to] = variables[from].copy_shallow();
        }
        
        public SetOfLocations getStaticField(jq_StaticField f) {
            if (TRACE_STATIC) out_ta.println("getting static field "+f+": "+static_variables.get(f));
            return (SetOfLocations)static_variables.get(f);
        }
        
        public void putStaticField(jq_StaticField f, SetOfLocations t) {
            if (TRACE_STATIC) out_ta.println("putting static field "+f+" with "+t+" old value "+static_variables.get(f));
            static_variables.put(f, t);
        }
        
        public AnalysisState copy_deep() {
            return new AnalysisState(this);
        }
        
        // Changes "this", but not "that"
        public boolean union_deep(AnalysisState that) {
            if (this == that) return false;
            boolean change = false;
            Assert._assert(this.parameters.length == that.parameters.length);
            HashMap old_to_new = new HashMap();
            Stack s = new Stack();
            for (int i=0; i<this.parameters.length; ++i) {
                if (that.parameters[i] == null) continue;
                Assert._assert(this.parameters[i] != null);
                if (this.parameters[i].union_deep(that.parameters[i], old_to_new, s)) change = true;
            }
            if (this.variables != null) {
                for (int i=0; i<this.variables.length; ++i) {
                    if (that.variables[i] == null) {
                        if (this.variables[i] != null) {
                            // not all paths have a reference in this variable
                            change = true; this.variables[i] = null;
                        }
                        continue;
                    }
                    if (this.variables[i] == null) {
                        // not all paths have a reference in this variable
                        continue;
                    }
                    if (this.variables[i].union_deep(that.variables[i], old_to_new, s)) change = true;
                }
            }
            for (Iterator i=that.static_variables.entrySet().iterator(); i.hasNext(); ) {
                Map.Entry e = (Map.Entry)i.next();
                SetOfLocations t2 = (SetOfLocations)e.getValue();
                SetOfLocations t1 = (SetOfLocations)this.static_variables.get(e.getKey());
                if (t1 == null) {
                    t1 = t2.copy_deep(old_to_new);
                    // add in static field value from outside
                    jq_StaticField sf = (jq_StaticField)e.getKey();
                    StaticFieldLocation sfl = new StaticFieldLocation(sf, sf.getType());
                    sfl.methodSequences = new MethodCallSequences(true);
                    t1.add(sfl);
                    this.static_variables.put(e.getKey(), t1);
                } else {
                    if (t1.union_deep(t2, old_to_new, s)) change = true;
                }
            }
            return change;
        }
        
        // netiher changes this nor that
        public AnalysisState merge_jsr(AnalysisState that) {
            Assert._assert(this.parameters.length == that.parameters.length);
            AnalysisState dis = this.copy_deep();
            HashMap old_to_new = new HashMap();
            Stack s = new Stack();
            for (int i=0; i<dis.parameters.length; ++i) {
                if (that.parameters[i] == null) continue;
                Assert._assert(dis.parameters[i] != null);
                dis.parameters[i].union_deep(that.parameters[i], old_to_new, s);
            }
            if (dis.variables != null) {
                for (int i=0; i<dis.variables.length; ++i) {
                    if (that.variables[i] == null) {
                        continue;
                    }
                    if (dis.variables[i] == null) {
                        dis.variables[i] = that.variables[i].copy_deep(old_to_new);
                        continue;
                    }
                    dis.variables[i].union_deep(that.variables[i], old_to_new, s);
                }
            }
            for (Iterator i=that.static_variables.entrySet().iterator(); i.hasNext(); ) {
                Map.Entry e = (Map.Entry)i.next();
                SetOfLocations t2 = (SetOfLocations)e.getValue();
                SetOfLocations t1 = (SetOfLocations)dis.static_variables.get(e.getKey());
                if (t1 == null) {
                    t1 = t2.copy_deep(old_to_new);
                    // add in static field value from outside
                    jq_StaticField sf = (jq_StaticField)e.getKey();
                    StaticFieldLocation sfl = new StaticFieldLocation(sf, sf.getType());
                    sfl.methodSequences = new MethodCallSequences(true);
                    t1.add(sfl);
                    dis.static_variables.put(e.getKey(), t1);
                } else {
                    t1.union_deep(t2, old_to_new, s);
                }
            }
            return dis;
        }
        
        public void dump() {
            HashSet visited = new HashSet();
            for (int i=0; i<parameters.length; ++i) {
                if (parameters[i] != null) {
                    out_ta.print(Strings.left("P"+i+":", 4));
                    String s = parameters[i].toString();
                    out_ta.print(s);
                    AnalysisSummary.dump_recurse(out_ta, visited, 4+s.length(), parameters[i]);
                }
            }
            if (static_variables != null) {
                Iterator it = static_variables.entrySet().iterator();
                while (it.hasNext()) {
                    Map.Entry e = (Map.Entry)it.next();
                    jq_StaticField d = (jq_StaticField)e.getKey();
                    SetOfLocations t = (SetOfLocations)e.getValue();
                    String s = d.getName().toString()+":";
                    out_ta.print(s);
                    AnalysisSummary.dump_recurse(out_ta, visited, s.length(), t);
                }
            }
            out_ta.println();
            if (variables != null) {
                out_ta.println("Variables ("+variables.length+" total)");
                for (int i=0; i<variables.length; ++i) {
                    if (variables[i] != null) {
                        out_ta.println("  "+i+": "+variables[i]);
                    }
                }
            }
        }
        
    }
    
    // A summary includes upward-exposed and downward-exposed calls for everything
    // reachable from the parameters and accessed static members, plus returned
    // and thrown members.
    public static class AnalysisSummary {
        jq_Method method;
        ParamLocation[] params;
        Map/*<jq_StaticField, SetOfLocations>*/ static_vars;

        ParamLocation[] params_ex;
        Map/*<jq_StaticField, SetOfLocations>*/ static_vars_ex;
        
        SetOfLocations returned;
        SetOfLocations thrown;
        
        AnalysisSummary(jq_Method method) {
            this.method = method;
            //this.params = new ParamLocation[method.getParamTypes().length];
            //this.params_ex = new ParamLocation[method.getParamTypes().length];
        }
        
        void addToReturned(SetOfLocations t) {
            HashMap m = new HashMap();
            if (returned == null) {
                returned = t.copy_deep(m);
            } else {
                returned.union_deep(t, m, new Stack());
            }
        }
        void addToThrown(SetOfLocations t) {
            HashMap m = new HashMap();
            if (thrown == null) {
                thrown = t.copy_deep(m);
            } else {
                thrown.union_deep(t, m, new Stack());
            }
        }
        
        public static final boolean TRIM_SUMMARIES = true;
        public static boolean TRACE_TRIM = false;
        
        void finish(AnalysisState last, AnalysisState ex_last) {
            if (last != null) {
                Assert._assert(last.parameters != null);
                this.params = last.parameters;
                this.static_vars = last.static_variables;
            } else {
                this.params = null;
                this.static_vars = null;
            }
            if (ex_last != null) {
                Assert._assert(ex_last.parameters != null);
                this.params_ex = ex_last.parameters;
                this.static_vars_ex = ex_last.static_variables;
            } else {
                this.params_ex = null;
                this.static_vars_ex = null;
            }
            
            if (TRIM_SUMMARIES) {
                if (TRACE_TRIM) {
                    out_ta.println("Summary before trimming:");
                    dump(false);
                }
                if (this.params != null) {
                    if (TRACE_TRIM) out_ta.println("Trimming normal exit...");
                    HashSet necessary = new HashSet();
                    HashSet visited = new HashSet();
                    if (returned != null)
                        for (Iterator i=this.returned.iterator(); i.hasNext(); ) necessary.add(i.next());
                    for (int i=0; i<this.params.length; ++i) {
                        if (this.params[i] == null) continue;
                        necessaryHelper(necessary, this.params[i], visited);
                    }
                    for (Iterator i=this.static_vars.entrySet().iterator(); i.hasNext(); ) {
                        Map.Entry e = (Map.Entry)i.next();
                        SetOfLocations s = (SetOfLocations)e.getValue();
                        Iterator j=s.iterator();
                        if (s.size() == 1) {
                            ProgramLocation pl = (ProgramLocation)j.next();
                            if ((!(pl instanceof StaticFieldLocation)) ||
                                (((StaticFieldLocation)pl).field != e.getKey())) {
                                necessary.add(pl);
                            }
                            necessaryHelper(necessary, pl, visited);
                        } else while (j.hasNext()) {
                            ProgramLocation pl = (ProgramLocation)j.next();
                            necessary.add(pl);
                            necessaryHelper(necessary, pl, visited);
                        }
                    }
                    if (TRACE_TRIM) out_ta.println("Necessary set: "+necessary);
                    visited = new HashSet();
                    for (int i=0; i<this.params.length; ++i) {
                        if (this.params[i] == null) continue;
                        visited.add(this.params[i]); trimHelper(this.params[i], necessary, visited);
                    }
                    for (Iterator i=this.static_vars.entrySet().iterator(); i.hasNext(); ) {
                        Map.Entry e = (Map.Entry)i.next();
                        SetOfLocations s = (SetOfLocations)e.getValue();
                        for (Iterator j=s.iterator(); j.hasNext(); ) {
                            ProgramLocation pl = (ProgramLocation)j.next();
                            if (!(pl instanceof OutsideProgramLocation)) continue;
                            OutsideProgramLocation opl = (OutsideProgramLocation)pl;
                            visited.add(opl);
                            if (trimHelper(opl, necessary, visited)) {
                                if (TRACE_TRIM) out_ta.println("static field node "+opl+" is useless, removing it");
                                j.remove();
                            }
                        }
                        if (s.size() == 0) {
                            if (TRACE_TRIM) out_ta.println("all edges from static field "+e.getKey()+" have been removed");
                            i.remove();
                        }
                    }
                }
                if (this.params_ex != null) {
                    if (TRACE_TRIM) out_ta.println("Trimming exceptional exit...");
                    HashSet necessary = new HashSet();
                    HashSet visited = new HashSet();
                    if (thrown != null)
                        for (Iterator i=this.thrown.iterator(); i.hasNext(); ) necessary.add(i.next());
                    for (int i=0; i<this.params_ex.length; ++i) {
                        if (this.params_ex[i] == null) continue;
                        visited.add(this.params_ex[i]); necessaryHelper(necessary, this.params_ex[i], visited);
                    }
                    for (Iterator i=this.static_vars_ex.entrySet().iterator(); i.hasNext(); ) {
                        Map.Entry e = (Map.Entry)i.next();
                        SetOfLocations s = (SetOfLocations)e.getValue();
                        Iterator j=s.iterator();
                        if (s.size() == 1) {
                            ProgramLocation pl = (ProgramLocation)j.next();
                            if ((!(pl instanceof StaticFieldLocation)) ||
                                (((StaticFieldLocation)pl).field != e.getKey())) {
                                necessary.add(pl);
                            }
                            necessaryHelper(necessary, pl, visited);
                        } else while (j.hasNext()) {
                            ProgramLocation pl = (ProgramLocation)j.next();
                            necessary.add(pl);
                            necessaryHelper(necessary, pl, visited);
                        }
                    }
                    if (TRACE_TRIM) out_ta.println("Necessary set: "+necessary);
                    visited = new HashSet();
                    for (int i=0; i<this.params_ex.length; ++i) {
                        if (this.params_ex[i] == null) continue;
                        visited.add(this.params_ex[i]); trimHelper(this.params_ex[i], necessary, visited);
                    }
                    for (Iterator i=this.static_vars_ex.entrySet().iterator(); i.hasNext(); ) {
                        Map.Entry e = (Map.Entry)i.next();
                        SetOfLocations s = (SetOfLocations)e.getValue();
                        for (Iterator j=s.iterator(); j.hasNext(); ) {
                            ProgramLocation pl = (ProgramLocation)j.next();
                            if (!(pl instanceof OutsideProgramLocation)) continue;
                            OutsideProgramLocation opl = (OutsideProgramLocation)pl;
                            visited.add(opl);
                            if (trimHelper(opl, necessary, visited)) {
                                if (TRACE_TRIM) out_ta.println("static field node "+opl+" is useless, removing it");
                                j.remove();
                            }
                        }
                        if (s.size() == 0) {
                            if (TRACE_TRIM) out_ta.println("all edges from static field "+e.getKey()+" have been removed");
                            i.remove();
                        }
                    }
                }
            }
        }
        private static boolean necessaryHelper(HashSet necessary, ProgramLocation pl, HashSet visited) {
            if (visited.contains(pl)) return necessary.contains(pl);
            visited.add(pl);
            if (pl instanceof OutsideProgramLocation) {
                Map outside = ((OutsideProgramLocation)pl).getOutsideEdges();
                if (outside != null) {
                    for (Iterator i=outside.entrySet().iterator(); i.hasNext(); ) {
                        Map.Entry e = (Map.Entry)i.next();
                        SetOfLocations s = (SetOfLocations)e.getValue();
                        for (Iterator j=s.iterator(); j.hasNext(); ) {
                            OutsideProgramLocation p = (OutsideProgramLocation)j.next();
                            if (necessaryHelper(necessary, p, visited)) necessary.add(pl);
                        }
                    }
                }
            }
            Map inside = pl.getInsideEdges();
            if (inside != null) {
                for (Iterator i=inside.entrySet().iterator(); i.hasNext(); ) {
                    Map.Entry e = (Map.Entry)i.next();
                    SetOfLocations s = (SetOfLocations)e.getValue();
                    for (Iterator j=s.iterator(); j.hasNext(); ) {
                        ProgramLocation p = (ProgramLocation)j.next();
                        necessary.add(p); necessary.add(pl);
                        necessaryHelper(necessary, p, visited);
                    }
                }
            }
            return necessary.contains(pl);
        }
        private boolean trimHelper(OutsideProgramLocation pl, HashSet necessary, HashSet visited) {
            Map outside = pl.getOutsideEdges();
            if (outside != null) {
                for (Iterator i=outside.entrySet().iterator(); i.hasNext(); ) {
                    Map.Entry e = (Map.Entry)i.next();
                    SetOfLocations s = (SetOfLocations)e.getValue();
                    MethodCallSequences new_mcs = new MethodCallSequences(false);
                    for (Iterator j=s.iterator(); j.hasNext(); ) {
                        OutsideProgramLocation p = (OutsideProgramLocation)j.next();
                        if (!visited.contains(p)) {
                            visited.add(p);
                            if (trimHelper(p, necessary, visited)) {
                                if (TRACE_TRIM) out_ta.println("node "+p+" is useless, removing it");
                                j.remove();
                            } else {
                                new_mcs.union_deep(p.methodSequences);
                            }
                        }
                    }
                    if (s.size() == 0) {
                        if (TRACE_TRIM) out_ta.println("all outside edges of "+pl+" field "+e.getKey()+" have been removed");
                        i.remove();
                    } else if (s.size() > 1) {
                        Dereference deref = (Dereference)e.getKey();
                        DereferenceLocation dl = new DereferenceLocation(deref, method, -1, deref.getType());
                        dl.methodSequences = new_mcs;
                        e.setValue(SetOfLocations.makeDerefSet(dl));
                        if (TRACE_TRIM) out_ta.println("made summary deref "+dl);
                    }
                }
            }
            Map inside = pl.getInsideEdges();
            MethodCallSequences mcs = pl.getMethodCallSequences();
            if (((outside == null) || (outside.size() == 0)) &&
                ((inside == null) || (inside.size() == 0)) &&
                mcs.isEmpty() &&
                !necessary.contains(pl)) {
                return true;
            } else {
                return false;
            }
        }
        
        boolean union_deep(AnalysisSummary that) {
            boolean change = false;
            HashMap old_to_new = new HashMap();
            Stack stack = new Stack();
            // make outside roots map to each other.
            if (that.params != null) {
                if (this.params == null) {
                    jq_Type[] paramTypes = method.getParamTypes();
                    this.params = new ParamLocation[paramTypes.length];
                    for (int i=0, j=0; i<paramTypes.length; ++i, ++j) {
                        jq_Type p = paramTypes[i];
                        if (p.isReferenceType()) {
                            ParamLocation pl = new ParamLocation(method, i, p);
                            pl.methodSequences = new MethodCallSequences(true);
                            this.params[i] = pl;
                        }
                        else if (p.getReferenceSize() == 8) ++j;
                    }
                }
                for (int i=0; i<that.params.length; ++i) {
                    if (that.params[i] == null) continue;
                    Assert._assert(this.params[i] != null);
                    old_to_new.put(IdentityHashCodeWrapper.create(that.params[i]), this.params[i]);
                }
            }
            if (that.params_ex != null) {
                if (this.params_ex == null) {
                    jq_Type[] paramTypes = method.getParamTypes();
                    this.params_ex = new ParamLocation[paramTypes.length];
                    for (int i=0, j=0; i<paramTypes.length; ++i, ++j) {
                        jq_Type p = paramTypes[i];
                        if (p.isReferenceType()) {
                            ParamLocation pl = new ParamLocation(method, i, p);
                            pl.methodSequences = new MethodCallSequences(true);
                            this.params_ex[i] = pl;
                        }
                        else if (p.getReferenceSize() == 8) ++j;
                    }
                }
                for (int i=0; i<that.params_ex.length; ++i) {
                    if (that.params_ex[i] == null) continue;
                    Assert._assert(this.params_ex[i] != null);
                    old_to_new.put(IdentityHashCodeWrapper.create(that.params_ex[i]), this.params_ex[i]);
                }
            }
            // no need to match static var locations -- they already match.
            
            // union params
            if (that.params != null) {
                Assert._assert(this.params != null);
                Assert._assert(this.params.length == that.params.length);
                for (int i=0; i<this.params.length; ++i) {
                    if (this.params[i] == null) {
                        Assert._assert(that.params[i] == null);
                        continue;
                    }
                    Assert._assert(that.params[i] != null);
                    if (this.params[i].union_deep(that.params[i], old_to_new, stack)) change = true;
                }
            }
            if (that.params_ex != null) {
                Assert._assert(this.params_ex != null);
                Assert._assert(this.params_ex.length == that.params_ex.length);
                for (int i=0; i<this.params_ex.length; ++i) {
                    if (this.params_ex[i] == null) {
                        Assert._assert(that.params_ex[i] == null);
                        continue;
                    }
                    Assert._assert(that.params_ex[i] != null);
                    if (this.params_ex[i].union_deep(that.params_ex[i], old_to_new, stack)) change = true;
                }
            }

            // union static vars
            if (that.static_vars != null) {
                if (this.static_vars == null)
                    this.static_vars = new HashMap();
                for (Iterator i=that.static_vars.entrySet().iterator(); i.hasNext(); ) {
                    Map.Entry e = (Map.Entry)i.next();
                    SetOfLocations t2 = (SetOfLocations)e.getValue();
                    SetOfLocations t1 = (SetOfLocations)this.static_vars.get(e.getKey());
                    if (t1 == null) {
                        this.static_vars.put(e.getKey(), t2.copy_deep(old_to_new));
                    } else {
                        t1.union_deep(t2, old_to_new, stack);
                    }
                }
            }
            if (that.static_vars_ex != null) {
                if (this.static_vars_ex == null)
                    this.static_vars_ex = new HashMap();
                for (Iterator i=that.static_vars_ex.entrySet().iterator(); i.hasNext(); ) {
                    Map.Entry e = (Map.Entry)i.next();
                    SetOfLocations t2 = (SetOfLocations)e.getValue();
                    SetOfLocations t1 = (SetOfLocations)this.static_vars_ex.get(e.getKey());
                    if (t1 == null) {
                        this.static_vars_ex.put(e.getKey(), t2.copy_deep(old_to_new));
                    } else {
                        t1.union_deep(t2, old_to_new, stack);
                    }
                }
            }
            // union returned set
            if (that.returned != null) {
                if (this.returned == null) {
                    this.returned = that.returned.copy_deep(old_to_new);
                } else {
                    this.returned.union_deep(that.returned, old_to_new, stack);
                }
            }
            // union thrown set
            if (that.thrown != null) {
                if (this.thrown == null) {
                    this.thrown = that.thrown.copy_deep(old_to_new);
                } else {
                    this.thrown.union_deep(that.thrown, old_to_new, stack);
                }
            }
            return change;
        }
        
        AnalysisSummary copy_deep() {
            AnalysisSummary that = new AnalysisSummary(this.method);
            HashMap old_to_new = new HashMap();
            if (this.params != null) {
                that.params = new ParamLocation[this.method.getParamTypes().length];
                for (int i=0; i<this.params.length; ++i) {
                    ParamLocation t = this.params[i];
                    if (t != null)
                        that.params[i] = (ParamLocation)t.copy_deep(old_to_new);
                }
            }
            if (this.params_ex != null) {
                that.params_ex = new ParamLocation[this.method.getParamTypes().length];
                for (int i=0; i<this.params_ex.length; ++i) {
                    ParamLocation t = this.params_ex[i];
                    if (t != null)
                        that.params_ex[i] = (ParamLocation)t.copy_deep(old_to_new);
                }
            }
            if (this.static_vars != null) {
                that.static_vars = new HashMap();
                for (Iterator i=this.static_vars.entrySet().iterator(); i.hasNext(); ) {
                    Map.Entry e = (Map.Entry)i.next();
                    jq_StaticField f = (jq_StaticField)e.getKey();
                    SetOfLocations t = (SetOfLocations)e.getValue();
                    Assert._assert(t != null);
                    that.static_vars.put(f, t.copy_deep(old_to_new));
                }
            }
            if (this.static_vars_ex != null) {
                that.static_vars_ex = new HashMap();
                for (Iterator i=this.static_vars_ex.entrySet().iterator(); i.hasNext(); ) {
                    Map.Entry e = (Map.Entry)i.next();
                    jq_StaticField f = (jq_StaticField)e.getKey();
                    SetOfLocations t = (SetOfLocations)e.getValue();
                    Assert._assert(t != null);
                    that.static_vars_ex.put(f, t.copy_deep(old_to_new));
                }
            }
            SetOfLocations t = this.returned;
            if (t != null)
                that.returned = t.copy_deep(old_to_new);
            t = this.thrown;
            if (t != null)
                that.thrown = t.copy_deep(old_to_new);
            return that;
        }
        
        boolean equivalent(AnalysisSummary that) {
            // TODO
            return true;
        }
        
        // Before: Already indented, pl has already been printed.
        // After:  Ends in newline.
        static void dump_recurse(java.io.PrintStream out, Set visited, int indent, ProgramLocation pl) {
            boolean b = false;
            if (pl.getInsideEdges() != null) {
                Iterator i = pl.getInsideEdges().entrySet().iterator();
                b = i.hasNext();
                while (i.hasNext()) {
                    Map.Entry e = (Map.Entry)i.next();
                    Dereference d = (Dereference)e.getKey();
                    SetOfLocations t2 = (SetOfLocations)e.getValue();
                    String s = "-" + d.toString() + "->";
                    out_ta.print(s);
                    dump_recurse(out, visited, indent+s.length(), t2);
                    if (i.hasNext()) 
                        out_ta.print(Strings.left("", indent));
                }
            }
            if (pl instanceof OutsideProgramLocation) {
                if (((OutsideProgramLocation)pl).getOutsideEdges() != null) {
                    Iterator i = ((OutsideProgramLocation)pl).getOutsideEdges().entrySet().iterator();
                    if (b && i.hasNext())
                        out_ta.print(Strings.left("", indent));
                    b |= i.hasNext();
                    while (i.hasNext()) {
                        Map.Entry e = (Map.Entry)i.next();
                        Dereference d = (Dereference)e.getKey();
                        SetOfLocations t2 = (SetOfLocations)e.getValue();
                        String s = "=" + d.toString() + "=>";
                        out_ta.print(s);
                        dump_recurse(out, visited, indent+s.length(), t2);
                        if (i.hasNext()) 
                            out_ta.print(Strings.left("", indent));
                    }
                }
            }
            if (!b) out_ta.println();
        }
        
        // Before: Already indented.
        // After:  Ends in newline.
        static void dump_recurse(java.io.PrintStream out, Set visited, int indent, SetOfLocations t) {
            if (visited.contains(t)) {
                out_ta.println("<backedge to "+Integer.toHexString(t.hashCode())+">");
                return;
            }
            visited.add(t);
            //out_ta.println(t.toString());
            out_ta.println(Integer.toHexString(t.hashCode()));
            Iterator j = t.sources.iterator();
            while (j.hasNext()) {
                ProgramLocation pl = (ProgramLocation)j.next();
                String s = pl.toString();
                out_ta.print(Strings.left("", indent));
                out_ta.print(s);
                dump_recurse(out, visited, indent+s.length(), pl);
            }
        }
        
        void dump(boolean ex) {
            java.io.PrintStream out = System.out;
            out_ta.println((ex?"Exception":"Regular")+" Summary for method "+method+":");
            HashSet visited = new HashSet();
            ParamLocation[] _params = (ex?params_ex:params);
            if (_params != null) {
                for (int i=0; i<_params.length; ++i) {
                    if (_params[i] != null) {
                        out_ta.print(Strings.left("P"+i+":", 4));
                        String s = _params[i].toString();
                        out_ta.print(s);
                        dump_recurse(out, visited, 4+s.length(), _params[i]);
                    }
                }
            }
            Map _static_vars = (ex?static_vars_ex:static_vars);
            if (_static_vars != null) {
                Iterator it = _static_vars.entrySet().iterator();
                while (it.hasNext()) {
                    Map.Entry e = (Map.Entry)it.next();
                    jq_StaticField d = (jq_StaticField)e.getKey();
                    SetOfLocations t = (SetOfLocations)e.getValue();
                    String s = d.getName().toString()+":";
                    out_ta.print(s);
                    dump_recurse(out, visited, s.length(), t);
                }
            }
            out_ta.println();
            if (returned != null) {
                String s = "Returned: ";
                out_ta.print(s);
                dump_recurse(out, visited, s.length(), returned);
            }
            if (thrown != null) {
                String s = "Thrown: ";
                out_ta.print(s);
                dump_recurse(out, visited, s.length(), thrown);
            }
            out_ta.println();
        }
        
    }
    
    public static class TypeAnalysisVisitor extends StackDepthVisitor {
        
        final int nParams, nLocals, nStack;
        AnalysisState currentState;
        final Map/*<ProgramLocation, DerefLocation>*/ loc_map;
        final AnalysisSummary summary;
        
        final AnalysisState[] in_states;
        final AnalysisState[] out_states;
        
        BasicBlock currentBB;
        boolean change;
        
        final Stack callStack; Set do_it_again;

        public static boolean ALWAYS_TRACE = false;
        public static HashSet trace_method_names = new HashSet();

        TypeAnalysisVisitor(jq_Method m,
                            ControlFlowGraph cfg,
                            Stack callStack,
                            Set do_it_again,
                            AnalysisState[] in_states,
                            AnalysisState[] out_states) {
            super(m, cfg);
            this.nParams = m.getParamWords();
            this.nLocals = m.getMaxLocals();
            this.nStack = m.getMaxStack();
            //this.currentState = new AnalysisState(m);
            this.loc_map = new HashMap();
            this.summary = new AnalysisSummary(m);
            this.callStack = callStack;
            this.do_it_again = do_it_again;
            this.TRACE = ALWAYS_TRACE;
            if (trace_method_names.contains(m.getName().toString())) {
                this.TRACE = true;
            }
            this.in_states = in_states;
            this.out_states = out_states;
        }
        
        public String toString() { return "TA/"+method.getName(); }
        
        boolean shouldRecordMethod(ProgramLocation receiverType, jq_Method f) {
            if (method instanceof jq_InstanceMethod) {
                if (receiverType.equals(currentState.parameters[0])) {
                    if (TRACE) out.println("Skipping recording call to "+f+" on this");
                    return false;
                }
            }
            //f.getDeclaringClass().load();
            //if (!f.getDeclaringClass().isInterface()) return false;
            /*** TEMPORARILY COMMENT OUT UNTIL WE CHECK IN DETOX STUFF. ***
            if (detox.runtime.Interceptor.crassFilter(f.getName().toString())) return false;
             *** TEMPORARILY COMMENT OUT UNTIL WE CHECK IN DETOX STUFF. ***/
            //if (TypeCheck.isAssignable(method.getDeclaringClass(), f.getDeclaringClass())) return false;
            //if (method.getDeclaringClass() == f.getDeclaringClass()) return false;
            return true;
        }
        
        public void visitBasicBlock(BasicBlock bb) {
            this.currentBB = bb;
            super.visitBasicBlock(bb);
        }
        
        public void visitACONST(Object s) {
            super.visitACONST(s);
            jq_Type type = null;
            if (s != null) type = Reflection.getTypeOf(s);
            SetOfLocations t = SetOfLocations.makeConstantSet(s, type, method, i_start);
            currentState.put(nLocals+currentStackDepth-1, t);
        }
        public void visitALOAD(int i) {
            currentState.move(nLocals+currentStackDepth, i);
            super.visitALOAD(i);
        }
        public void visitISTORE(int i) {
            super.visitISTORE(i);
            currentState.put(i, null);
        }
        public void visitFSTORE(int i) {
            super.visitFSTORE(i);
            currentState.put(i, null);
        }
        public void visitLSTORE(int i) {
            super.visitLSTORE(i);
            currentState.put(i, null);
            currentState.put(i+1, null);
        }
        public void visitDSTORE(int i) {
            super.visitDSTORE(i);
            currentState.put(i, null);
            currentState.put(i+1, null);
        }
        public void visitASTORE(int i) {
            currentState.move(i, nLocals+currentStackDepth-1);
            super.visitASTORE(i);
        }
        public void visitIALOAD() {
            currentState.put(nLocals+currentStackDepth-2, null);
            super.visitIALOAD();
        }
        public void visitFALOAD() {
            currentState.put(nLocals+currentStackDepth-2, null);
            super.visitFALOAD();
        }
        public void visitLALOAD() {
            currentState.put(nLocals+currentStackDepth-2, null);
            super.visitLALOAD();
        }
        public void visitDALOAD() {
            currentState.put(nLocals+currentStackDepth-2, null);
            super.visitDALOAD();
        }
        public void visitAALOAD() {
            SetOfLocations base = currentState.get(nLocals+currentStackDepth-2);
            super.visitAALOAD();
            // look up array stores off the base
            jq_Array arrayType;
            jq_Type basetype = base.getType();
            if (basetype instanceof jq_Array) arrayType = (jq_Array)basetype;
            else arrayType = PrimordialClassLoader.getJavaLangObject().getArrayTypeForElementType();
            ArrayDereference d = new ArrayDereference(arrayType);
            SetOfLocations deref = base.dereference(d, method, i_start);
            currentState.put(nLocals+currentStackDepth-1, deref);
        }
        public void visitAASTORE() {
            SetOfLocations base = currentState.get(nLocals+currentStackDepth-3);
            SetOfLocations obj = currentState.get(nLocals+currentStackDepth-1);
            currentState.put(nLocals+currentStackDepth-3, null);
            currentState.put(nLocals+currentStackDepth-1, null);
            super.visitAASTORE();
            jq_Array arrayType;
            jq_Type basetype = base.getType();
            if (basetype instanceof jq_Array) arrayType = (jq_Array)basetype;
            else arrayType = PrimordialClassLoader.getJavaLangObject().getArrayTypeForElementType();
            ArrayDereference d = new ArrayDereference(arrayType);
            base.store(d, obj);
        }
        public void visitARETURN() {
            SetOfLocations t = currentState.get(nLocals+currentStackDepth-1);
            currentState.put(nLocals+currentStackDepth-1, null);
            summary.addToReturned(t);
            super.visitARETURN();
        }
        public void visitAGETSTATIC(jq_StaticField f) {
            super.visitAGETSTATIC(f);
            // look up what has been stored off of this static field.
            SetOfLocations t = currentState.getStaticField(f);
            if (t == null) {
                // no writes to this static field.
                if (TRACE) out.println("No writes to static field "+f+" yet");
                t = SetOfLocations.makeStaticFieldSet(f);
                currentState.putStaticField(f, t);
            } else {
                // already had a write to this static field.
                if (TRACE) out.println("Writes to static field "+f+": "+t);
            }
            currentState.put(nLocals+currentStackDepth-1, t);
        }
        public void visitAPUTSTATIC(jq_StaticField f) {
            super.visitAPUTSTATIC(f);
            SetOfLocations t = currentState.get(nLocals+currentStackDepth);
            currentState.put(nLocals+currentStackDepth, null);
            // look up what has been stored off of this static field.
            SetOfLocations t2 = currentState.getStaticField(f);
            if (t2 == null) {
                // first write to this static field.
                if (TRACE) out.println("First write to static field "+f+": "+t);
                // strong update!
            } else {
                // already had a write to this static field.
                if (TRACE) out.println("Already had write to static field "+f+": "+t2+"! replacing with "+t);
                // strong update!
            }
            currentState.putStaticField(f, t);
        }
        public void visitAGETFIELD(jq_InstanceField f) {
            super.visitAGETFIELD(f);
            SetOfLocations base = currentState.get(nLocals+currentStackDepth-1);
            // look up field stores off the base
            FieldDereference d = new FieldDereference(f);
            SetOfLocations deref = base.dereference(d, method, i_start);
            currentState.put(nLocals+currentStackDepth-1, deref);
        }
        public void visitAPUTFIELD(jq_InstanceField f) {
            SetOfLocations base = currentState.get(nLocals+currentStackDepth-2);
            SetOfLocations obj = currentState.get(nLocals+currentStackDepth-1);
            currentState.put(nLocals+currentStackDepth-2, null);
            currentState.put(nLocals+currentStackDepth-1, null);
            super.visitAPUTFIELD(f);
            FieldDereference d = new FieldDereference(f);
            base.store(d, obj);
        }
        void visitInvoke(byte op, jq_Method f) {
            CallTargets targets;
            if (op != INVOKE_STATIC) {
                // look up "this" ptr
                SetOfLocations t = currentState.get(nLocals+currentStackDepth-f.getReturnWords());
                Iterator i = t.iterator();
                while (i.hasNext()) {
                    ProgramLocation source = (ProgramLocation)i.next();
                    if (shouldRecordMethod(source, f)) {
                        // set last called to this method.
                        String loc = this.method+":"+(int)this.i_start;
                        MethodCall mc = new MethodCall(this.method, this.i_start, f);
                        source.mustCall(mc, loc);
                    }
                }
                i = t.iterator();
                targets = CallTargets.NoCallTarget.INSTANCE;
                while (i.hasNext()) {
                    ProgramLocation source = (ProgramLocation)i.next();
                    jq_Reference rtype = (jq_Reference)source.getType();
                    CallTargets targets2;
                    if (rtype == null)
                        // TODO: attach to null ptr exception handler
                        targets2 = CallTargets.NoCallTarget.INSTANCE;
                    else
                        targets2 = CallTargets.getTargets(method.getDeclaringClass(), f, op, rtype, !source.isFromOutside(), true);
                    targets = targets.union(targets2);
                }
            } else {
                targets = CallTargets.getStaticTargets((jq_StaticMethod)f);
            }
            // look up and apply mappings
            if (TRACE) out.println("Method call targets: "+targets);
            Iterator i = targets.iterator();
            Set results = new LinearSet();
            while (i.hasNext()) {
                jq_Method target = (jq_Method)i.next();
                jq_Class klass = target.getDeclaringClass();
                klass.prepare();
                if (target.isNative()) {
                    if (TRACE) out.println("Native method call! from "+method+":"+(int)i_start+" to "+target);
                    // native method!!
                    continue;
                }
                AnalysisSummary r = (AnalysisSummary)summaries.get(target);
                if (callStack.contains(target)) {
                    // recursive method! skip it and mark it for iteration
                    if (TRACE) out.println("Call stack contains "+target+"! doing interprocedural iteration.");
                    do_it_again.add(target);
                    if (r != null) {
                        results.add(r);
                    }
                } else {
                    if (do_it_again.contains(target)) {
                        if (TRACE) out.println("Performing interprocedural iteration on "+target);
                        // call into a recursive cycle that needs iteration
                        Assert._assert(r != null);
                        for (;;) {
                            AnalysisSummary q = TypeAnalysis.analyze(target, callStack, do_it_again);
                            if (r.equivalent(q)) {
                                // finished iterating the recursive cycle
                                if (TRACE) out.println("Summaries are equivalent! Finished iterating "+target);
                                summaries.put(target, r);
                                do_it_again.remove(target);
                                break;
                            } else {
                                if (TRACE) out.println("Summaries are different! Continuing iteration of "+target);
                                Assert._assert(do_it_again.contains(target));
                            }
                        }
                    }
                    if (r == null) {
                        if (callStack.size() >= MAX_DEPTH) {
                            if (TRACE) out.println("Exceeded maximum depth, skipping call target "+target);
                            continue;
                        }
                        if (BY_PACKAGE && !method.getDeclaringClass().isInSamePackage(target.getDeclaringClass())) {
                            if (TRACE) out.println("Call target "+target+" is not in the same package as "+method+", skipping.");
                            continue;
                        }
                        if ((classesToAnalyze != null) && !classesToAnalyze.contains(target.getDeclaringClass())) {
                            if (TRACE) out.println("Call target "+target+" is not in the set of classes to analyze, skipping.");
                            continue;
                        }
                        r = analyze(target, callStack, do_it_again);
                        Assert._assert(r != null);
                        summaries.put(target, r);
                        if (TRACE) out.println("Cached result of analysis of method "+target+":");
                    }
                    results.add(r);
                }
            }
            if (results.size() == 0) {
                if (TRACE) out.println("No analysis results!");
                // use a seed for the return value.
                jq_Type returnType = f.getReturnType();
                if (returnType.isReferenceType()) {
                    SetOfLocations seed = SetOfLocations.makeSeed(returnType);
                    currentState.put(nLocals+currentStackDepth-1, seed);
                }
                CaughtLocation cl = new CaughtLocation(this.method, this.i_start, PrimordialClassLoader.getJavaLangThrowable());
                cl.methodSequences = new MethodCallSequences(true);
                SetOfLocations thrown = new SetOfLocations();
                thrown.add(cl);
                summary.addToThrown(thrown);
                if (TRACE) out.println("Call can throw "+thrown+", unioning with exception exit");
                if (in_states[0] == null) {
                    if (TRACE) out.println("First exceptional path, copying set");
                    in_states[0] = currentState.copy_deep();
                } else {
                    in_states[0].union_deep(currentState);
                }
                ExceptionHandlerIterator handleri = this.currentBB.getExceptionHandlers();
                while (handleri.hasNext()) {
                    ExceptionHandler handler = handleri.nextEH();
                    unionWithExceptionHandler(thrown, handler);
                }
                return;
            }
            AnalysisSummary final_result;
            if (results.size() == 1) {
                final_result = ((AnalysisSummary)results.iterator().next())/*.copy_deep()*/;
            } else {
                i = results.iterator();
                final_result = ((AnalysisSummary)i.next()).copy_deep();
                while (i.hasNext()) {
                    final_result.union_deep((AnalysisSummary)i.next());
                }
            }
            AnalysisState state_copy = currentState.copy_deep();
            SetOfLocations thrown = applySummary(currentState, f, final_result, false);

            thrown = applySummary(state_copy, f, final_result, true);
            CaughtLocation cl = new CaughtLocation(this.method, this.i_start, PrimordialClassLoader.getJavaLangThrowable());
            cl.methodSequences = new MethodCallSequences(true);
            if (thrown == null) {
                thrown = new SetOfLocations();
            }
            thrown.add(cl);
            summary.addToThrown(thrown);
            if (TRACE) out.println("Call can throw "+thrown+", unioning with exception exit");
            if (in_states[0] == null) {
                if (TRACE) out.println("First exceptional path, copying set");
                in_states[0] = state_copy.copy_deep();
            } else {
                in_states[0].union_deep(state_copy);
            }
            ExceptionHandlerIterator handleri = this.currentBB.getExceptionHandlers();
            while (handleri.hasNext()) {
                ExceptionHandler handler = handleri.nextEH();
                unionWithExceptionHandler(thrown, handler);
            }
        }

        void unionWithExceptionHandler(SetOfLocations thrown, ExceptionHandler ex) {
            SetOfLocations[] a = new SetOfLocations[method.getMaxStack()];
            BasicBlock bb2 = ex.getEntry();
            SetOfLocations caught = thrown.filterByType(ex.getExceptionType());
            if (caught.size() == 0) return;
            if (TRACE) out.println("Exceptions "+caught+" can go to handler "+ex);
            int i = method.getMaxLocals();
            System.arraycopy(currentState.variables, i, a, 0, a.length);
            currentState.variables[i] = caught;
            for (++i; i<a.length; ++i) {
                currentState.variables[i] = null;
            }
            if (in_states[bb2.id] != null) {
                Assert._assert(bb2.startingStackDepth == 1);
                if (in_states[bb2.id].union_deep(currentState)) {
                    if (TRACE) out.println("In set for exception handler "+bb2+" changed!");
                    this.change = true;
                }
            } else {
                if (TRACE) out.println("No in set for exception handler "+bb2+" yet");
                in_states[bb2.id] = currentState.copy_deep();
                this.change = true;
                bb2.startingStackDepth = 1;
            }
            System.arraycopy(a, 0, currentState.variables, method.getMaxLocals(), a.length);
        }

        public static int MAX_DEPTH = 8;
        public static boolean BY_PACKAGE = false;
        
        static boolean isInMultimap(HashMap map, ProgramLocation t1, SetOfLocations t2) {
            SetOfLocations s = (SetOfLocations)map.get(t1);
            if (s == null) return false;
            else {
                for (Iterator i = t2.iterator(); i.hasNext(); ) {
                    if (!s.contains((ProgramLocation)i.next())) return false;
                }
                return true;
            }
        }
        
        static boolean addToMultimap(HashMap map, ProgramLocation t1, SetOfLocations t2) {
            SetOfLocations s = (SetOfLocations)map.get(t1);
            if (s == null) {
                map.put(t1, s = SetOfLocations.makeEmptySet());
            } else {
                // aliasing!
            }
            return s.addAll(t2);
        }
        
        SetOfLocations mapCalleeSetIntoCaller(HashMap old_to_new,
                                              HashMap callee_to_caller_map_out,
                                              HashMap callee_to_caller_map_in,
                                              SetOfLocations inside_callee_edge) {
            if (TRACE) out.println("Mapping callee set "+inside_callee_edge+" to caller");
            Stack stack = new Stack();
            SetOfLocations result = SetOfLocations.makeEmptySet();
            for (Iterator i=inside_callee_edge.iterator(); i.hasNext(); ) {
                ProgramLocation callee_node = (ProgramLocation)i.next();
                if (callee_node instanceof OutsideProgramLocation) {
                    SetOfLocations caller_node_set = (SetOfLocations)callee_to_caller_map_out.get(callee_node);
                    if (TRACE) out.println("Outside callee node "+callee_node+" corresponds to caller set "+caller_node_set);
                    if (caller_node_set == null) {
                        // the callee loaded from an uninitialized field of a captured node in the caller
                        // TODO: add "null" to the result set.
                    } else {
                        // if a cast occurred on this outside node, need to filter by type
                        OutsideProgramLocation dl = (OutsideProgramLocation)callee_node;
                        if (dl.getOriginalType() != dl.getType()) {
                            if (TRACE) out.println("Cast occurred on callee outside node "+dl+" to type "+dl.getType()+", filtering our nodes "+caller_node_set);
                            caller_node_set = caller_node_set.filterByType(dl.getType());
                        }
                        result.union_deep(caller_node_set, old_to_new, stack);
                    }
                } else {
                    SetOfLocations caller_node_set = (SetOfLocations)callee_to_caller_map_in.get(callee_node);
                    if (caller_node_set == null) {
                        caller_node_set = SetOfLocations.makeEmptySet();
                        caller_node_set.add(callee_node.copy_deep(old_to_new));
                        callee_to_caller_map_in.put(callee_node, caller_node_set);
                    }
                    if (TRACE) out.println("Inside callee node "+callee_node+" corresponds to caller node "+caller_node_set);
                    result.union_deep(caller_node_set, old_to_new, stack);
                }
            }
            if (TRACE) out.println("Result of mapping callee set "+inside_callee_edge+" to caller: "+result);
            return result;
        }
        
        public SetOfLocations applySummary(AnalysisState state, jq_Method f, AnalysisSummary summary, boolean ex) {
            Assert._assert(summary != null);
            if (TRACE) {
                out.println(" --------======> APPLYING "+(ex?"EXCEPTION":"")+" SUMMARY "+f+":");
                summary.dump(ex);
                out.println(" --------======> TO STATE");
                state.dump();
            }
            jq_Type[] paramTypes = f.getParamTypes();
            HashMap callee_to_caller_mmap = new HashMap();
            Stack callee_node_worklist = new Stack();
            Assert._assert(summary.params != null);
            ParamLocation[] summary_params;
            if (!ex) summary_params = summary.params;
            else summary_params = summary.params_ex;
            if (summary_params != null) {
                for (int i=0, j=0; i<paramTypes.length; ++i, ++j) {
                    jq_Type paramType = paramTypes[i];
                    if (paramType.isPrimitiveType()) {
                        if (paramType.getReferenceSize() == 8) ++j;
                        continue;
                    }
                    SetOfLocations t_caller = state.get(nLocals+currentStackDepth-f.getReturnWords()+j);
                    state.put(nLocals+currentStackDepth-f.getReturnWords()+j, null);
                    ParamLocation p_callee = summary_params[i];
                    if (TRACE) out.println("Param "+i+" Callee outside node "+p_callee+" matches caller set "+t_caller);
                    addToMultimap(callee_to_caller_mmap, p_callee, t_caller);
                    Assert._assert(!callee_node_worklist.contains(p_callee));
                    callee_node_worklist.push(p_callee);
                }
            }
            Map summary_static_vars;
            if (!ex) summary_static_vars = summary.static_vars;
            else summary_static_vars = summary.static_vars_ex;
            if (summary_static_vars != null) {
                Iterator it = summary_static_vars.entrySet().iterator();
                while (it.hasNext()) {
                    Map.Entry e = (Map.Entry)it.next();
                    jq_StaticField sf = (jq_StaticField)e.getKey();
                    SetOfLocations static_var_set = (SetOfLocations)e.getValue();
                    Iterator it2 = static_var_set.iterator();
                    while (it2.hasNext()) {
                        ProgramLocation pl = (ProgramLocation)it2.next();
                        if (pl instanceof StaticFieldLocation) {
                            StaticFieldLocation sfl = (StaticFieldLocation)pl;
                            if (sf != sfl.field) continue;
                            SetOfLocations t_caller2 = (SetOfLocations)state.static_variables.get(sfl.field);
                            if (t_caller2 == null) {
                                // callee has a static node that caller doesn't have
                                if (TRACE) out.println("Callee has static field "+sfl.field+" that caller doesn't have: "+sfl);
                                state.static_variables.put(sfl.field, t_caller2 = SetOfLocations.makeStaticFieldSet(sfl.field));
                            } else {
                                if (TRACE) out.println("Callee has static field "+sfl.field+" as node "+sfl+" that matches caller set "+t_caller2);
                            }
                            addToMultimap(callee_to_caller_mmap, sfl, t_caller2);
                            if (callee_node_worklist.contains(sfl)) {
                                out.println(method+" ERROR!! callee node worklist ("+callee_node_worklist+") already contains sf node: "+sfl);
                                Assert.UNREACHABLE();
                            }
                            callee_node_worklist.push(sfl);
                        }
                    }
                }
            }
            HashMap old_to_new = new HashMap();
            while (!callee_node_worklist.empty()) {
                // Get outside node from worklist
                OutsideProgramLocation outside_node = (OutsideProgramLocation)callee_node_worklist.pop();
                if (outside_node.getOutsideEdges() == null) continue;
                // Get the set of inside nodes that match this outside node
                SetOfLocations inside_node_set = (SetOfLocations)callee_to_caller_mmap.get(outside_node);
                if (TRACE) out.println("Visiting outside edges of node "+outside_node+", matches caller set "+inside_node_set);
                Assert._assert(inside_node_set != null);
                //jq.Assert(inside_node_set.size() > 0);
                // Iterate through the set of inside nodes, matching outside edges to inside edges
                Iterator it = outside_node.getOutsideEdges().entrySet().iterator();
                while (it.hasNext()) {
                    Map.Entry e = (Map.Entry)it.next();
                    Dereference deref = (Dereference)e.getKey();
                    SetOfLocations outside_node_outside_edge_set = (SetOfLocations)e.getValue();
                    SetOfLocations inside_node_inside_edge_set = inside_node_set.dereference(deref, outside_node_outside_edge_set, old_to_new);
                    Iterator it3 = outside_node_outside_edge_set.iterator();
                    while (it3.hasNext()) {
                        DereferenceLocation outside_edge = (DereferenceLocation)it3.next();
                        // if a cast occurred on this outside node, need to filter by type
                        if (outside_edge.deref.getType() != outside_edge.getType()) {
                            if (TRACE) out.println("Cast occurred on callee outside node "+outside_edge+" to type "+outside_edge.getType()+", filtering our nodes "+inside_node_inside_edge_set);
                            inside_node_inside_edge_set = inside_node_inside_edge_set.filterByType(outside_edge.getType());
                        }
                        if (addToMultimap(callee_to_caller_mmap, outside_edge, inside_node_inside_edge_set)) {
                            if (TRACE) out.println("Set changed for callee outside node "+outside_edge+" after adding "+inside_node_inside_edge_set+", adding to worklist");
                            callee_node_worklist.push(outside_edge);
                        }
                    }
                }
            }
            // mapping is complete. add inside edges on the callee's outside nodes to the caller
            Iterator it = callee_to_caller_mmap.entrySet().iterator();
            HashMap callee_to_caller_map_in = new HashMap();
            while (it.hasNext()) {
                Map.Entry e = (Map.Entry)it.next();
                OutsideProgramLocation outside_callee = (OutsideProgramLocation)e.getKey();
                SetOfLocations inside_caller_set = (SetOfLocations)e.getValue();
                if (outside_callee.getInsideEdges() == null) continue;
                if (TRACE) out.println("Adding inside edges of callee outside node "+outside_callee+" to caller set "+inside_caller_set);
                Iterator edges = outside_callee.getInsideEdges().entrySet().iterator();
                while (edges.hasNext()) {
                    Map.Entry e2 = (Map.Entry)edges.next();
                    Dereference d = (Dereference)e2.getKey();
                    SetOfLocations inside_callee_edge = (SetOfLocations)e2.getValue();
                    SetOfLocations inside_caller_edge = mapCalleeSetIntoCaller(old_to_new, callee_to_caller_mmap, callee_to_caller_map_in, inside_callee_edge);
                    inside_caller_set.store(d, inside_caller_edge);
                }
            }
            // apply method call order on the callee's outside nodes to the caller's nodes
            it = callee_to_caller_mmap.entrySet().iterator();
            HashMap caller_to_method_seq = new HashMap();
            if (TRACE) out.println("Adding callee method call sequences...");
            while (it.hasNext()) {
                Map.Entry e = (Map.Entry)it.next();
                ProgramLocation outside_callee = (ProgramLocation)e.getKey();
                MethodCallSequences callee_seq = outside_callee.getMethodCallSequences();
                if (callee_seq.isEmpty()) continue;
                SetOfLocations inside_caller_set = (SetOfLocations)e.getValue();
                if (TRACE) out.println("Adding callee sequence "+callee_seq+" on caller nodes "+inside_caller_set);
                for (Iterator it2 = inside_caller_set.iterator(); it2.hasNext(); ) {
                    ProgramLocation inside_caller = (ProgramLocation)it2.next();
                    MethodCallSequences seq = (MethodCallSequences)caller_to_method_seq.get(inside_caller);
                    if (seq == null) {
                        caller_to_method_seq.put(inside_caller, seq = callee_seq.copy_deep());
                        if (TRACE) out.println("First method call sequence on "+inside_caller+": "+seq);
                    } else {
                        if (TRACE) out.println("Intersecting method call sequence on "+inside_caller+": "+callee_seq);
                        seq.intersect_deep(callee_seq);
                        if (TRACE) out.println("Result of intersection: "+seq);
                    }
                }
            }
            it = caller_to_method_seq.entrySet().iterator();
            while (it.hasNext()) {
                Map.Entry e = (Map.Entry)it.next();
                ProgramLocation caller = (ProgramLocation)e.getKey();
                MethodCallSequences seq = (MethodCallSequences)e.getValue();
                if (ex) {
                    if (TRACE) out.println("Merging to exceptional exit method sequence "+seq+" for node "+caller);
                    caller.methodSequences.union_deep(seq);
                } else {
                    if (TRACE) out.println("Applying first called method sets "+seq.printFirstCalled()+" to node "+caller);
                    String loc = "Method call at "+this.method+":"+(int)this.i_start;
                    caller.addFirstCalledToMayCall(seq, loc);
                    caller.updateMustCall(seq);
                }
            }
            if (!ex) {
                if (summary.returned != null) {
                    SetOfLocations caller_node_set = mapCalleeSetIntoCaller(old_to_new, callee_to_caller_mmap, callee_to_caller_map_in, summary.returned);
                    state.put(nLocals+currentStackDepth-1, caller_node_set);
                } else {
                    if (f.getReturnType().isReferenceType()) {
                        // method has no return (must end in athrow)
                        SetOfLocations seed = SetOfLocations.makeSeed(f.getReturnType());
                        currentState.put(nLocals+currentStackDepth-1, seed);
                    }
                }
                if (summary.thrown != null) {
                    SetOfLocations caller_node_set = mapCalleeSetIntoCaller(old_to_new, callee_to_caller_mmap, callee_to_caller_map_in, summary.thrown);
                    return caller_node_set;
                }
            }
            return null;
        }

        public void visitIINVOKE(byte op, jq_Method f) {
            super.visitIINVOKE(op, f);
            visitInvoke(op, f);
        }
        public void visitLINVOKE(byte op, jq_Method f) {
            super.visitLINVOKE(op, f);
            visitInvoke(op, f);
        }
        public void visitFINVOKE(byte op, jq_Method f) {
            super.visitFINVOKE(op, f);
            visitInvoke(op, f);
        }
        public void visitDINVOKE(byte op, jq_Method f) {
            super.visitDINVOKE(op, f);
            visitInvoke(op, f);
        }
        public void visitAINVOKE(byte op, jq_Method f) {
            super.visitAINVOKE(op, f);
            visitInvoke(op, f);
        }
        public void visitVINVOKE(byte op, jq_Method f) {
            super.visitVINVOKE(op, f);
            visitInvoke(op, f);
        }
        public void visitNEW(jq_Type f) {
            super.visitNEW(f);
            SetOfLocations t = SetOfLocations.makeNewSet(f, method, i_start);
            currentState.put(nLocals+currentStackDepth-1, t);
        }
        public void visitNEWARRAY(jq_Array f) {
            super.visitNEWARRAY(f);
            SetOfLocations t = SetOfLocations.makeNewSet(f, method, i_start);
            currentState.put(nLocals+currentStackDepth-1, t);
        }
        public void visitCHECKCAST(jq_Type f) {
            super.visitCHECKCAST(f);
            SetOfLocations t = currentState.get(nLocals+currentStackDepth-1);
            SetOfLocations t2 = t.filterByType(f);
            // todo: check for class cast exceptions
            currentState.put(nLocals+currentStackDepth-1, t2);
        }
        public void visitATHROW() {
            SetOfLocations t = currentState.get(nLocals+currentStackDepth-1);
            currentState.put(nLocals+currentStackDepth-1, null);
            summary.addToThrown(t);
            if (in_states[0] == null) {
                if (TRACE) out.println("First exceptional path, copying set");
                in_states[0] = currentState.copy_deep();
            } else {
                in_states[0].union_deep(currentState);
            }
            dontMergeWithSuccessors = true;
            super.visitATHROW();
        }
        public void visitMONITOR(byte op) {
            super.visitMONITOR(op);
        }
        public void visitMULTINEWARRAY(jq_Type f, char dim) {
            super.visitMULTINEWARRAY(f, dim);
            SetOfLocations t = SetOfLocations.makeNewSet(f, method, i_start);
            currentState.put(nLocals+currentStackDepth-1, t);
        }
        public void visitPOP() {
            super.visitPOP();
            currentState.put(nLocals+currentStackDepth, null);
        }
        public void visitPOP2() {
            super.visitPOP2();
            currentState.put(nLocals+currentStackDepth, null);
            currentState.put(nLocals+currentStackDepth+1, null);
        }
        public void visitDUP() {
            SetOfLocations t1 = currentState.get(nLocals+currentStackDepth-1);
            currentState.put(nLocals+currentStackDepth, t1);
            super.visitDUP();
        }
        public void visitDUP_x1() {
            SetOfLocations t1 = currentState.get(nLocals+currentStackDepth-1);
            SetOfLocations t2 = currentState.get(nLocals+currentStackDepth-2);
            currentState.put(nLocals+currentStackDepth-2, t1);
            currentState.put(nLocals+currentStackDepth-1, t2);
            currentState.put(nLocals+currentStackDepth, t1);
            super.visitDUP_x1();
        }
        public void visitDUP_x2() {
            SetOfLocations t1 = currentState.get(nLocals+currentStackDepth-1);
            SetOfLocations t2 = currentState.get(nLocals+currentStackDepth-2);
            SetOfLocations t3 = currentState.get(nLocals+currentStackDepth-3);
            currentState.put(nLocals+currentStackDepth-3, t1);
            currentState.put(nLocals+currentStackDepth-2, t3);
            currentState.put(nLocals+currentStackDepth-1, t2);
            currentState.put(nLocals+currentStackDepth, t1);
            super.visitDUP_x2();
        }
        public void visitDUP2() {
            SetOfLocations t1 = currentState.get(nLocals+currentStackDepth-1);
            SetOfLocations t2 = currentState.get(nLocals+currentStackDepth-2);
            currentState.put(nLocals+currentStackDepth, t2);
            currentState.put(nLocals+currentStackDepth+1, t1);
            super.visitDUP2();
        }
        public void visitDUP2_x1() {
            SetOfLocations t1 = currentState.get(nLocals+currentStackDepth-1);
            SetOfLocations t2 = currentState.get(nLocals+currentStackDepth-2);
            SetOfLocations t3 = currentState.get(nLocals+currentStackDepth-3);
            currentState.put(nLocals+currentStackDepth-3, t2);
            currentState.put(nLocals+currentStackDepth-2, t1);
            currentState.put(nLocals+currentStackDepth-1, t3);
            currentState.put(nLocals+currentStackDepth, t2);
            currentState.put(nLocals+currentStackDepth+1, t1);
            super.visitDUP2_x1();
        }
        public void visitDUP2_x2() {
            SetOfLocations t1 = currentState.get(nLocals+currentStackDepth-1);
            SetOfLocations t2 = currentState.get(nLocals+currentStackDepth-2);
            SetOfLocations t3 = currentState.get(nLocals+currentStackDepth-3);
            SetOfLocations t4 = currentState.get(nLocals+currentStackDepth-4);
            currentState.put(nLocals+currentStackDepth-4, t2);
            currentState.put(nLocals+currentStackDepth-3, t1);
            currentState.put(nLocals+currentStackDepth-2, t4);
            currentState.put(nLocals+currentStackDepth-1, t3);
            currentState.put(nLocals+currentStackDepth, t2);
            currentState.put(nLocals+currentStackDepth+1, t1);
            super.visitDUP2_x2();
        }
        public void visitSWAP() {
            SetOfLocations t1 = currentState.get(nLocals+currentStackDepth-1);
            SetOfLocations t2 = currentState.get(nLocals+currentStackDepth-2);
            currentState.put(nLocals+currentStackDepth-2, t1);
            currentState.put(nLocals+currentStackDepth-1, t2);
            super.visitSWAP();
        }
        
        public void visitJSR(int target) {
            Assert._assert(currentBB.getNumberOfSuccessors() == 1);
            BasicBlock jsub_bb = currentBB.getSuccessor(0);
            SetOfLocations jsub = SetOfLocations.makeJSRSubroutine(jsub_bb);
            if (TRACE) out.println("Made JSR subroutine from "+currentBB+" to target "+jsub_bb);
            currentState.put(nLocals+currentStackDepth, jsub);
            super.visitJSR(target);
        }
        boolean dontMergeWithSuccessors = false;
        public void visitRET(int loc) {
            super.visitRET(loc);
            dontMergeWithSuccessors = true;
            SetOfLocations s = currentState.get(loc);
            Iterator it = s.iterator();
            while (it.hasNext()) {
                JSRSubroutine jsub = (JSRSubroutine)it.next();
                BasicBlock jsub_bb = jsub.jsub_bb;
                if (currentBB.getNumberOfSuccessors() == 0) {
                    if (TRACE) out.println("Adding jsr subroutine edges to "+currentBB);
                    change = true;
                    currentBB.setSubroutineRet(cfg, jsub_bb);
                    if (TRACE) {
                        out.println("Number of jsr subroutine edges: "+currentBB.getNumberOfSuccessors());
                        for (int j=0; j<currentBB.getNumberOfSuccessors(); ++j) {
                            out.println("Successor "+j+": "+currentBB.getSuccessor(j));
                        }
                    }
                }
                // for each BB that calls into the JSR subroutine, union the current state
                // and the out state of the BB into the successor BB.
                // note: union is special in that nulls in vars in the current state don't
                // destroy data in the BB state.
                for (int i=0; i<jsub_bb.getNumberOfPredecessors(); ++i) {
                    BasicBlock bb = jsub_bb.getPredecessor(i);
                    AnalysisState out_state = out_states[bb.id];
                    if (TRACE) out.println("Merging jsr subroutine info from "+currentBB+" and "+bb+" into "+cfg.getBasicBlock(bb.id+1));
                    if (out_state == null) {
                        if (TRACE) out.println(bb+" has not been analyzed yet!");
                        continue;
                    }
                    in_states[bb.id+1] = out_state.merge_jsr(currentState);
                    cfg.getBasicBlock(bb.id+1).startingStackDepth = currentStackDepth;
                }
            }
        }
    }
    
    static class JSRSubroutine extends ProgramLocation {
        BasicBlock jsub_bb;
        JSRSubroutine(BasicBlock jsub_bb) { this.jsub_bb = jsub_bb; }
        public ProgramLocation filterByType(jq_Type type) { return this; }
        public jq_Type getType() { return null; }
        boolean union_deep(ProgramLocation that, HashMap old_to_new, Stack stack) {
            Assert._assert(((JSRSubroutine)that).jsub_bb == this.jsub_bb);
            return false;
        }
        public ProgramLocation copy_deep(HashMap old_to_new) {
            JSRSubroutine that = (JSRSubroutine)old_to_new.get(IdentityHashCodeWrapper.create(this));
            if (that == null) {
                if (TRACE_COPY) out_ta.println("New location to copy deep: "+this);
                that = new JSRSubroutine(this.jsub_bb);
                old_to_new.put(IdentityHashCodeWrapper.create(this), that);
            }
            return that;
        }
        public ProgramLocation copy_shallow(HashMap old_to_new) {
            JSRSubroutine that = (JSRSubroutine)old_to_new.get(IdentityHashCodeWrapper.create(this));
            if (that == null) {
                if (TRACE_COPY) out_ta.println("New location to copy shallow: "+this);
                that = new JSRSubroutine(this.jsub_bb);
                old_to_new.put(IdentityHashCodeWrapper.create(this), that);
            }
            return that;
        }
        public boolean equals(JSRSubroutine that) {
            return (jsub_bb == that.jsub_bb);
        }
        public boolean equals(Object o) {
            if (o instanceof JSRSubroutine) return equals((JSRSubroutine)o);
            return false;
        }
        public int hashCode() { return jsub_bb.hashCode(); }
        public String toString() { return "jsrsub:"+jsub_bb; }
    }

    public static class ScalarReplacementVisitor extends TypeAnalysisVisitor {
        
        ScalarReplacementVisitor(jq_Method m,
                                 ControlFlowGraph cfg,
                                 Stack callStack,
                                 Set do_it_again,
                                 AnalysisState[] in_states,
                                 AnalysisState[] out_states) {
            super(m, cfg, callStack, do_it_again, in_states, out_states);
        }
        
        public String toString() { return "SR/"+method.getName(); }
        
    }

}
