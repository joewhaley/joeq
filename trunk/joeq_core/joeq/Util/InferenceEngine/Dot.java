/*
 * Created on May 10, 2004
 *
 * To change the template for this generated file go to
 * Window - Preferences - Java - Code Generation - Code and Comments
 */
package joeq.Util.InferenceEngine;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;

import joeq.Util.Assert;
import joeq.Util.Collections.HashWorklist;
import joeq.Util.Collections.Worklist;
import joeq.Util.Graphs.Graph;
import joeq.Util.IO.MyStringTokenizer;
import joeq.Util.InferenceEngine.RelationGraph.GraphNode;

/**
 * @author cunkel
 *
 * To change the template for this generated type comment go to
 * Window - Preferences - Java - Code Generation - Code and Comments
 */
public class Dot {
    private Solver solver;
    private Collection/*EdgeSource*/ edgeSources;
    private Collection/*NodeAttributeModifier*/ nodeModifiers;
    private Worklist worklist;
    private Collection/*String*/ nodes;
    private Collection/*String*/ edges;
    private Collection/*Relation*/ usedRelations;
    private String outputFileName;
    
    Dot() {
        edgeSources = new LinkedList();
        nodeModifiers = new LinkedList();
        worklist = new HashWorklist(true);
        nodes = new LinkedList();
        edges = new LinkedList();
        usedRelations = new HashSet();
    }

    void outputGraph() throws IOException {
        Iterator i = edgeSources.iterator();
        Set allRoots = new HashSet();
        while (i.hasNext()) {
            allRoots.addAll(((EdgeSource) i.next()).roots());
        }
        i = allRoots.iterator();
        while (i.hasNext()) {
            worklist.push(i.next());
        }
        
        while (!worklist.isEmpty()) {
            GraphNode n = (GraphNode) worklist.pull();
            visitNode(n);
        }
        
        DataOutputStream dos = null;
        try {
            dos = new DataOutputStream(new FileOutputStream(outputFileName));
            
            dos.writeBytes("digraph {\n");
            dos.writeBytes("  size=\"10,7.5\";\n");
            dos.writeBytes("  rotate=90;\n");
            dos.writeBytes("  concentrate=true;\n");
            dos.writeBytes("  ratio=fill;\n");
            dos.writeBytes("\n");
            
            i = nodes.iterator();
            while (i.hasNext()) {
                dos.writeBytes("  ");
                dos.writeBytes((String) i.next());
            }
            dos.writeBytes("\n");
            
            i = edges.iterator();
            while (i.hasNext()) {
                dos.writeBytes("  ");
                dos.writeBytes((String) i.next());
            }
            dos.writeBytes("}\n");
        }
        finally {
            if (dos != null) { dos.close(); }
        }
    }
    
    void parseInput(Solver s, LineNumberReader in) throws IOException {
        solver = s;
        String currentLine = in.readLine();
        while (currentLine != null) {
            System.out.println("Parsing "+currentLine);
            MyStringTokenizer st = new MyStringTokenizer(currentLine, " ,()");
            parseLine(st);
            currentLine = in.readLine();
        }
    }
    
    private void parseLine(MyStringTokenizer st) {
        if (!st.hasMoreTokens()) {
            return;
        }
        String s = st.nextToken();
        if (s.startsWith("#")) {
            return;
        }
        if (s.equals("edge")) {
            String relationName = st.nextToken();
            String roots = st.nextToken();
            String rootsName = st.nextToken();
            if (!roots.equals("roots")) {
                throw new IllegalArgumentException();
            }
            
            Relation edgeRelation = solver.getRelation(relationName);
            Relation rootsRelation = solver.getRelation(rootsName);
            
            usedRelations.add(edgeRelation);
            usedRelations.add(rootsRelation);

            EdgeSource es = new EdgeSource(edgeRelation,
                                           rootsRelation);
            
            if (st.hasMoreTokens()) {
                String label = st.nextToken();
                if (!label.equals("label")) {
                    throw new IllegalArgumentException();
                }
                relationName = st.nextToken();
                String openParen = st.nextToken();
                String [] e = new String[3];
                e[0] = st.nextToken();
                String comma = st.nextToken();
                e[1] = st.nextToken();
                String comma2 = st.nextToken();
                e[2] = st.nextToken();
                String closeParen = st.nextToken();
                if (!openParen.equals("(")) throw new IllegalArgumentException();
                if (!comma.equals(",")) throw new IllegalArgumentException();
                if (!comma2.equals(",")) throw new IllegalArgumentException();
                if (!closeParen.equals(")")) throw new IllegalArgumentException();
                int sourceIndex = -1;
                int labelIndex = -1;
                int sinkIndex = -1;
                for (int i=0; i<3; i++) {
                    if (e[i].equals("source")) {
                        sourceIndex = i;
                    }
                    else if (e[i].equals("sink")) {
                        sinkIndex = i;
                    }
                    else if (e[i].equals("label")) {
                        labelIndex = i;
                    }
                }
                if (sourceIndex == -1) throw new IllegalArgumentException();
                if (sinkIndex == -1) throw new IllegalArgumentException();
                if (labelIndex == -1) throw new IllegalArgumentException();
                                
                es.setLabelSource(new LabelSource((BDDRelation)solver.getRelation(relationName), sourceIndex, sinkIndex, labelIndex));
            }
            edgeSources.add(es);
        }
        else if (s.equals("domain")) {
            String domainName = st.nextToken();
            String attribute = st.nextToken();
            String value = st.nextToken();
            
            nodeModifiers.add(new DomainModifier(attribute, value, solver.getFieldDomain(domainName)));
        }
        else if (s.equals("default")) {
            String attribute = st.nextToken();
            String value = st.nextToken();
            
            nodeModifiers.add(new DefaultModifier(attribute, value));
        }
        else if (s.equals("relation")) {
            String relationName = st.nextToken();
            String attribute = st.nextToken();
            String value = st.nextToken();
            
            BDDRelation relation = (BDDRelation) solver.getRelation(relationName);
            usedRelations.add(relation);
            nodeModifiers.add(new InRelationModifier(attribute, value, relation));
        }
        else if (s.equals("output")) {
            outputFileName = st.nextToken();
        }
        else {
            throw new IllegalArgumentException();
        }
    }
    
    
    private static class LabelSource {
        BDDRelation relation;
        int sourceIndex;
        int sinkIndex;
        int labelIndex;
        LabelSource(BDDRelation r, int sourceI, int sinkI, int labelI) {
            relation = r;
            sourceIndex = sourceI;
            sinkIndex = sinkI;
            labelIndex = labelI;
        }
        
        String getLabel(RelationGraph.GraphNode source, RelationGraph.GraphNode sink) {
            return "";
        }
    }
    
    
    private static class EdgeSource {
        Relation relation;
        Relation roots;
        LabelSource labelSource;
        
        Graph g;
        
        EdgeSource(Relation rel, Relation rts) {
            relation = rel;
            roots = rts;
            labelSource = null;
            g = null;
        }
        
        public void setLabelSource(LabelSource source) {
            labelSource = source;
        }
        
        public Collection roots() {
            if (g == null) {
                g = new RelationGraph(roots, relation);
            }
            
            return g.getRoots(); 
        }
        
        public void visitSources(Dot dot, RelationGraph.GraphNode sink, boolean addEdges) {
            if (g == null) {
                g = new RelationGraph(roots, relation);
            }
            
            Collection c = g.getNavigator().prev(sink);
            
            Iterator i = c.iterator();
            while (i.hasNext()) {
                RelationGraph.GraphNode source = (RelationGraph.GraphNode)i.next();
                
                dot.enqueue(source);
                
                if (addEdges) {
                    String label;
                    if (labelSource != null) {
                        dot.addEdge(dot.nodeName(source) + " -> " + dot.nodeName(sink) + 
                                    "[" + labelSource.getLabel(source, sink) + "];\n");
                    } else {
                        dot.addEdge(dot.nodeName(source) + " -> " + dot.nodeName(sink) + ";\n");
                    }
                        
                }
            }
        }
        
        public void visitSinks(Dot dot, RelationGraph.GraphNode source, boolean addEdges) {
            if (g == null) {
                g = new RelationGraph(roots, relation);
            }
            
            Collection c = g.getNavigator().next(source);
            
            Iterator i = c.iterator();
            while (i.hasNext()) {
                RelationGraph.GraphNode sink = (RelationGraph.GraphNode)i.next();
                
                dot.enqueue(sink);
                
                if (addEdges) {
                    String label;
                    if (labelSource != null) {
                        dot.addEdge(dot.nodeName(source) + " -> " + dot.nodeName(sink) + 
                                    "[" + labelSource.getLabel(source, sink) + "];\n");
                    } else {
                        dot.addEdge(dot.nodeName(source) + " -> " + dot.nodeName(sink) + ";\n");
                    }
                }
            }
        }
    }

    private static abstract class NodeAttributeModifier {
        abstract boolean match(RelationGraph.GraphNode n, Map a);    
    }
    
    private class DefaultModifier extends NodeAttributeModifier {
        String property;
        String value;
        DefaultModifier(String p, String v) {
            property = p;
            value = v;
        }

        boolean match(GraphNode n, Map a) {
            a.put(property, value);
            return true;
        }     
    }
    
    private class DomainModifier extends NodeAttributeModifier {
        FieldDomain fd;
        String property;
        String value;
        
        DomainModifier(String p, String v, FieldDomain f) {
            property = p;
            value = v;
            fd = f;
        }
        
        boolean match(GraphNode n, Map a) {
            FieldDomain f = n.v.getFieldDomain();
            
            if (f.equals(fd)) {
                a.put(property,value);
                return true;
            }
            else {
                return false;
            }
        }        
    }
    
    private class InRelationModifier extends NodeAttributeModifier {
        BDDRelation relation;
        String property;
        String value;
        
        InRelationModifier(String p, String v, BDDRelation r) {
            property = p;
            value = v;
            relation = r;
            
            Assert._assert(r.fieldDomains.size()==1);
        }

        boolean match(GraphNode n, Map a) {
            FieldDomain f = (FieldDomain) relation.fieldDomains.iterator().next();
            if (n.v.getFieldDomain().equals(f)) {
                if (relation.contains(0, n.number)) {
                    a.put(property,value);
                    return true;
                }
            }
            return false;
        }        
    }

    public void enqueue(GraphNode x) {
        worklist.push(x);
    }

    public void addEdge(String edge) {
        edges.add(edge);
    }

    public String nodeName(GraphNode n) {
        return "\"" + n.toString() + "\"";
    }
    
    private void visitNode (GraphNode x) {
        Map attributes = new HashMap();
        String nodeName = (String) x.v.getFieldDomain().map.get((int)x.number);
        if (nodeName != null) {
            attributes.put("label", nodeName);
        }
        Iterator i = nodeModifiers.iterator();
        while (i.hasNext()) {
            NodeAttributeModifier m = (NodeAttributeModifier) i.next();
            m.match(x, attributes);
        }
        String node = nodeName(x) + " [";
        i = attributes.keySet().iterator();
        boolean firstAttribute = true;
        while (i.hasNext()) {
            String attribute = (String)i.next();
            String value = (String)attributes.get(attribute);
            if (!firstAttribute) {
                node += ", ";
            }
            node += attribute + "=" + "\"" + value + "\"";
            firstAttribute = false;
        }
        node += "];\n";
        nodes.add(node);
        i = edgeSources.iterator();
        while (i.hasNext()) {
            EdgeSource es = (EdgeSource) i.next();
            es.visitSinks(this, x, true);
            //es.visitSources(this, x, false);
        }
    }

    /**
     * @return
     */
    public Collection getUsedRelations() {
        return usedRelations;
    }
}



