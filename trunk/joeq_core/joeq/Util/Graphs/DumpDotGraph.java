// DumpDotGraph.java, created May 17, 2004 2:51:42 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.Graphs;

import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import joeq.Util.Collections.HashWorklist;
import joeq.Util.Collections.IndexMap;
import joeq.Util.Collections.FilterIterator.Filter;

/**
 * DumpDotGraph
 * 
 * @author jwhaley
 * @version $Id$
 */
public class DumpDotGraph {
    
    // Graph nodes/edges
    Set nodes;
    Navigator navigator;
    
    // Graph labels
    Filter nodeLabels;
    EdgeLabeler edgeLabels;
    
    // Graph colors
    Filter nodeColors;
    EdgeLabeler edgeColors;
    
    // Graph styles
    Filter nodeStyles;
    EdgeLabeler edgeStyles;
    
    // Graph options
    boolean concentrate;
    
    // Clusters
    Filter containingCluster;
    Set clusters;
    Navigator clusterNavigator;
    
    public DumpDotGraph() {
        
    }
    
    public void setNavigator(Navigator navigator) {
        this.navigator = navigator;
    }
    
    public void setNodeLabels(Filter nodeLabels) {
        this.nodeLabels = nodeLabels;
    }
    
    public void setEdgeLabels(EdgeLabeler edgeLabels) {
        this.edgeLabels = edgeLabels;
    }
    
    public void setNodeColors(Filter nodeColors) {
        this.nodeColors = nodeColors;
    }
    
    public void setEdgeColors(EdgeLabeler edgeColors) {
        this.edgeColors = edgeColors;
    }
    
    public void setNodeStyles(Filter nodeStyles) {
        this.nodeStyles = nodeStyles;
    }
    
    public void setEdgeStyles(EdgeLabeler edgeStyles) {
        this.edgeStyles = edgeStyles;
    }
    
    public void setClusters(Filter clusters) {
        this.containingCluster = clusters;
    }
    
    public void setClusterNesting(Navigator nesting) {
        this.clusterNavigator = nesting;
    }
    
    public void setNodeSet(Set nodes) {
        this.nodes = nodes;
    }
    
    public Set computeTransitiveClosure(Collection roots) {
        HashWorklist w = new HashWorklist(true);
        w.addAll(roots);
        while (!w.isEmpty()) {
            Object o = w.pull();
            w.addAll(navigator.next(o));
        }
        nodes = w.getVisitedSet();
        return nodes;
    }
    
    public Set computeBidirTransitiveClosure(Collection roots) {
        HashWorklist w = new HashWorklist(true);
        w.addAll(roots);
        while (!w.isEmpty()) {
            Object o = w.pull();
            w.addAll(navigator.next(o));
            w.addAll(navigator.prev(o));
        }
        nodes = w.getVisitedSet();
        return nodes;
    }
    
    void computeClusters() {
        if (containingCluster == null) return;
        clusters = new HashSet();
        for (Iterator i = nodes.iterator(); i.hasNext(); ) {
            Object o = i.next();
            Object c = containingCluster.map(o);
            if (c != null) clusters.add(c);
        }
    }
    
    public void dump(String filename) throws IOException {
        DataOutputStream dos = null;
        try {
            dos = new DataOutputStream(new FileOutputStream(filename));
            dump(dos);
        } finally {
            if (dos != null) dos.close();
        }
    }
    
    void dumpNodes(DataOutput dos, IndexMap m, Object cluster) throws IOException {
        for (Iterator i = nodes.iterator(); i.hasNext(); ) {
            Object o = i.next();
            if (containingCluster != null) {
                Object c = containingCluster.map(o);
                if (c == cluster || !c.equals(cluster)) continue;
            }
            Object nodeid = (m != null) ? (Object)("n"+m.get(o)) : o;
            dos.writeBytes("  "+nodeid);
            boolean open = false;
            if (nodeLabels != null) {
                Object label = nodeLabels.map(o);
                if (label != null) {
                    open = true;
                    dos.writeBytes(" [label=\""+label+"\"");
                }
            }
            if (nodeStyles != null) {
                Object label = nodeStyles.map(o);
                if (label != null) {
                    if (!open) dos.writeBytes(" [");
                    else dos.writeBytes(",");
                    open = true;
                    dos.writeBytes("style="+label);
                }
            }
            if (nodeColors != null) {
                Object label = nodeColors.map(o);
                if (label != null) {
                    if (!open) dos.writeBytes(" [");
                    else dos.writeBytes(",");
                    open = true;
                    dos.writeBytes("color="+label);
                }
            }
            if (open) dos.writeBytes("]");
            
            dos.writeBytes(";\n");
        }
    }
    
    void dumpCluster(DataOutput dos, IndexMap m, Set visitedClusters, Object c) throws IOException {
        if (visitedClusters.add(c)) return;
        dos.writeBytes("  subgraph cluster"+visitedClusters.size()+" {\n");
        dumpNodes(dos, m, c);
        if (clusterNavigator != null) {
            Collection subClusters = clusterNavigator.next(c);
            for (Iterator i = subClusters.iterator(); i.hasNext(); ) {
                Object subC = i.next();
                dumpCluster(dos, m, visitedClusters, subC);
            }
        }
        dos.writeBytes("  }\n");
    }
    
    public void dump(DataOutput dos) throws IOException {
        computeClusters();
        
        dos.writeBytes("digraph {\n");
        dos.writeBytes("  size=\"10,7.5\";\n");
        dos.writeBytes("  rotate=90;\n");
        if (concentrate)
            dos.writeBytes("  concentrate=true;\n");
        dos.writeBytes("  ratio=fill;\n");
        dos.writeBytes("\n");
        
        IndexMap m;
        if (nodeLabels != null) {
            m = new IndexMap("NodeID");
        } else {
            m = null;
        }
        
        if (clusters != null) {
            Set visitedClusters = new HashSet();
            for (Iterator i = clusters.iterator(); i.hasNext(); ) {
                Object c = i.next();
                dumpCluster(dos, m, visitedClusters, c);
            }
        }
        dumpNodes(dos, m, null);
        
        for (Iterator i = nodes.iterator(); i.hasNext(); ) {
            Object n1 = i.next();
            Object node1id = (m != null) ? (Object)("n"+m.get(n1)) : n1;
            Collection succ = navigator.next(n1);
            for (Iterator j = succ.iterator(); j.hasNext(); ) {
                Object n2 = j.next();
                if (!nodes.contains(n2)) continue;
                Object node2id = (m != null) ? (Object)("n"+m.get(n2)) : n2;
                dos.writeBytes("  "+node1id+" -> "+node2id);
                boolean open = false;
                if (edgeLabels != null) {
                    Object label = edgeLabels.getLabel(n1, n2);
                    if (label != null) {
                        open = true;
                        dos.writeBytes(" [label=\""+label+"\"]");
                    }
                }
                if (edgeStyles != null) {
                    Object label = edgeStyles.getLabel(n1, n2);
                    if (label != null) {
                        if (!open) dos.writeBytes(" [");
                        else dos.writeBytes(",");
                        open = true;
                        dos.writeBytes("style="+label);
                    }
                }
                if (edgeColors != null) {
                    Object label = edgeColors.getLabel(n1, n2);
                    if (label != null) {
                        if (!open) dos.writeBytes(" [");
                        else dos.writeBytes(",");
                        open = true;
                        dos.writeBytes("color="+label);
                    }
                }
                dos.writeBytes(";\n");
            }
        }
        
        dos.writeBytes("}\n");
    }
    
}
