package Compil3r.Analysis.IPA;

import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Set;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.TypedBDDFactory.TypedBDD;

import Clazz.jq_Method;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.Node;
import Util.Assert;

/**
 * A helper class for PAResults.
 * @see PAResults
 * */
public class PAResultSelector {
    private PAResults _results;
    private PA r;
    private boolean _verbose = false;

    public PAResultSelector(PAResults results){
        _results = results;
        r = results.r;    
    }
    
    public TypedBDD getReturnBDD(jq_Method method) {
        // lookup in MRet
        int idx = r.Mmap.get(method);
        BDD i   = r.M.ithVar(idx);
        BDD restricted = r.Mret.restrictWith(i);
        
        //System.out.println("In getReturnBDD: " + _results.toString((TypedBDD) restricted, -1));
        
        return (TypedBDD)restricted.replaceWith(r.bdd.makePair(r.V2, r.V1));
    }
    
    public TypedBDD getFormalParamBDD(jq_Method method, int index) {
        // lookup in MRet
        int idx = r.Mmap.get(method);
        BDD i   = r.M.ithVar(idx);
    
        // formal is in MxZxV1
        return (TypedBDD)r.formal.restrictWith(i).restrictWith(r.Z.ithVar(index));
    }
    
    public BDD addAllContextToVar(BDD bdd) {
        TypedBDD tbdd = (TypedBDD)bdd;
        Set domains = tbdd.getDomainSet();

        Assert._assert(domains.size() == 1);
        BDDDomain dom = (BDDDomain)domains.iterator().next();
        Assert._assert(dom != r.V1c && dom != r.V2c);        
        
        tbdd.setDomains(dom, r.V1c);
        //BDD result = (TypedBDD)tbd q  d.and(r.V1c.set());
        //tbdd.free();
        
        //return result;
        return tbdd;
    }

    protected Collection getUses(TypedBDD bdd) {
        bdd = (TypedBDD)addAllContextToVar(bdd);
        TypedBDD reach = (TypedBDD)_results.calculateDefUse(bdd);
        BDD vars = reach.exist(r.V1c.set());
        
        Collection result = new LinkedList();
        for(Iterator iter = ( (TypedBDD)vars ).iterator(); iter.hasNext(); ) {
            TypedBDD var = (TypedBDD)iter.next();
            
            result.add(getNode(var));
        }
        return result; 
    }
    
    protected Collection getUses(Node node) {
        TypedBDD bdd = (TypedBDD)r.V1.ithVar(_results.getVariableIndex(node));
        
        return getUses(bdd);
    }
    
    public Node getNode(TypedBDD var) {              
        long[] indeces = r.V1.getVarIndices(var);
        Assert._assert(indeces.length == 1, "There are " + indeces.length + " indeces in " + var.toStringWithDomains());
        long index = indeces[0];        
        Node node = _results.getVariableNode((int)index);
        
        return node;
    }
    
    public void collectReachabilityTraces(TypedBDD start, TypedBDD stop) {
        /*
        try {
            //_results.printDefUseChain(bdd.andWith(r.V1c.set()));
            _results.defUseGraph(bdd.andWith(r.V1c.set()), true, new DataOutputStream(System.out));
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
        */
        
        LinkedList results = new LinkedList();
        Node root = getNode(start);
        Node sink = getNode(stop);
        System.out.println("Starting with " + root + ", looking for " + sink);
        collectReachabilityTraces(new PAReachabilityTrace(), root, sink, results);
        
        int i = 0;
        for(Iterator iter = results.iterator(); iter.hasNext(); i++) {
            PAReachabilityTrace trace = (PAReachabilityTrace)iter.next();
            
            System.out.println("Trace " + i + ": " + trace.toStringLong());
        }           
    }
    
    protected void collectReachabilityTraces(PAReachabilityTrace trace, Node last, Node stop, Collection results) {
        if(trace.contains(last)){
            if(_verbose) System.err.println("Already seen " + last + " in the trace " + trace.toString());
            results.add(trace); return; 
        }
        trace.addNode(last);
        if(stop == last) {
            if(_verbose) System.err.println("Found " + stop);
            results.add(trace); return;
        }
        Collection c = getUses(last);
        if(c.isEmpty()) {
            if(_verbose) System.err.println("Finished with " + last);
            results.add(trace); return;
        }
                   
        if(_verbose) {
            System.err.println("Node " + last + " has " + c.size() + " successor(s): " + c +
                        "\t trace: " + trace);
        }
        
        for(Iterator iter = c.iterator(); iter.hasNext();) {
            Node node = (Node)iter.next();
            
            PAReachabilityTrace newTrace = (PAReachabilityTrace)trace.clone();
            collectReachabilityTraces(newTrace, node, stop, results);
        }
    }
    
    class PAReachabilityTrace {
        LinkedList/*Node*/ _nodes;
        
        PAReachabilityTrace(){
            _nodes = new LinkedList();
        }
        public boolean contains(Node node) {
            return _nodes.contains(node);
        }
        void addNode(Node node) {
            _nodes.addLast(node);
        }
        public String toString() {
            StringBuffer buf = new StringBuffer(size() + " [");
            int i = 1;
            for(Iterator iter = _nodes.iterator(); iter.hasNext(); i++) {
                Node node = (Node)iter.next();
                
                buf.append(" (" + i + ")");
                buf.append(node.toString_short());                
            }
            buf.append("]");
            return buf.toString();
        }
        public String toStringLong() {
            StringBuffer buf = new StringBuffer(size() + " [\n");
            int i = 1;
            for(Iterator iter = _nodes.iterator(); iter.hasNext(); i++) {
                Node node = (Node)iter.next();
        
                buf.append("\t(" + i + ")");
                buf.append(node.toString_long());
                buf.append("\n");                
            }
            buf.append("]");
            return buf.toString();
        }
        public Object clone() {
            PAReachabilityTrace result = new PAReachabilityTrace();
            for(Iterator iter = _nodes.iterator(); iter.hasNext(); ) {
                result.addNode((Node)iter.next());
            } 
            Assert._assert(size() == result.size());
            return result;
        }
        int size() {return _nodes.size();}
    }

    public void collectReachabilityTraces2(BDD bdd) {
        bdd = addAllContextToVar(bdd);
        int i = 0;
        BDD reach;
        Assert._assert(_results != null);
        do {
            BDD vars = bdd.exist(r.V1c.set());
            System.err.print("Generation " + i + ": ");
            for(Iterator li = ((TypedBDD)vars).iterator(); li.hasNext(); ) {
                System.err.print(((TypedBDD)li.next()).toStringWithDomains() + " ");
            }
            System.err.print("\n");
                        
            // ((TypedBDD)bdd).satCount()
            //System.err.println("Generation " + i + ": " + bdd.toStringWithDomains() /*_pa.toString((TypedBDD)bdd, -1)*/ + "\n");
            reach = _results.calculateDefUse(bdd);
            bdd = reach;
            i++;                
        } while(i < 10 && !bdd.isZero());        
    }
}