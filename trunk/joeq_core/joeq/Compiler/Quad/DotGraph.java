// DotGraph.java, created Tue Nov  5 14:16:40 2002 by joewhaley
// Copyright (C) 2001-3 Godmar Back <gback@cs.utah.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Quad;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashSet;
import java.util.Iterator;

import Clazz.jq_Class;
import Clazz.jq_Method;
import Util.Templates.ListIterator;

/**
 * @author Godmar Back <gback@cs.utah.edu, @stanford.edu> 
 *
 * This class is a ControlFlowGraphVisitor.
 * For each CFG, it produces a "dot" file in the output directory. 
 * See or change createMethodName to adapt how the filenames are formed.
 *
 * @see DotGraph#outputDir
 * @see DotGraph#dotFilePrefix
 * @see DotGraph#createMethodName
 *
 * @version $Id$
 */
public class DotGraph implements ControlFlowGraphVisitor {

    /**
     * The output directory for the dot graph descriptions
     */
    public static String outputDir = "dottedcfgs";

    /**
     * Prefix that goes before the name.
     */
    public static String dotFilePrefix = "joeq-";

    /**
     * Adapt this method to create filenames the way you want them.
     */
    String createMethodName(jq_Method mth) {
	String filename = dotFilePrefix + mth.toString();
	filename = filename.replace('/', '_');
	filename = filename.replace(' ', '_');
	filename = filename.replace('<', '_');
	filename = filename.replace('>', '_');
	return filename;
    }

    /**
     * dot - helper class for outputting graphviz specifications for simple cfgs
     *
     * See http://www.research.att.com/sw/tools/graphviz/
     *
     * Process with, for instance, "dot -Tgif -o graph.gif <inputfile>"
     * or simply "dotgif <inputfile>"
     *
     * @author Godmar Back <gback@cs.utah.edu>
     */
    public static class dot {
        private static PrintWriter containedgraph = null;

        public static void openGraph(String name) {
            try {
                String dirname = outputDir;
                File d = new File(dirname);
                if (!d.exists()) {
                    d.mkdir();
                }
                String dirsep = System.getProperty("file.separator");
                containedgraph = new PrintWriter(new FileOutputStream(dirname + dirsep + name));
                containedgraph.println("digraph contained_in_graph {");
                containedgraph.println("\tnode[shape=box,fontname = \"Arial\", fontsize=10];");
                containedgraph.println("\tedge[fontname = \"Arial\", fontcolor=red, fontsize=8];");
            } catch (IOException _) {
                _.printStackTrace(System.err);
            }
        }

        public static String escape(String from) {
            from = from.replace('\t', ' ').trim();
            StringBuffer fb = new StringBuffer();
            for (int i = 0, sucspaces = 0; i < from.length(); i++) {
                char c = from.charAt(i);
                if (c == '"' || c == '\\')
                    fb.append("\\" + c);
                else if (c == '\n')
                    fb.append("\\\\n");
                else if (c == '\r')
                    fb.append("\\\\r");
                else if (sucspaces == 0 || c != ' ')
                    fb.append(c);
                if (c == ' ')
                    sucspaces++;
                else
                    sucspaces = 0;
            }
            return fb.toString();
        }

        private static void outputString(String from) {
            containedgraph.print("\"" + escape(from) + "\"");
        }

        private static void labelEdge(String edge) {
            if (edge != null) {
                containedgraph.print("[label=");
                outputString(edge);
                containedgraph.print(",color=red]");
            }
        }

        public static void userDefined(String useroutput) {
            if (containedgraph != null) {
                containedgraph.print(useroutput);
            }
        }

        private static void makeCircleNode(String to) {
            containedgraph.print("\t");
            outputString(to);
            containedgraph.println("[shape=circle,fontcolor=red,color=red];");
        }

        public static void addEntryEdge(String from, String to, String edge) {
            if (containedgraph != null) {
                makeCircleNode(from);
                containedgraph.print("\t");
                outputString(from);
                containedgraph.print(" -> ");
                outputString(to);
                labelEdge(edge);
                containedgraph.println(";");
            }
        }

        public static void addLeavingEdge(String from, String to, String edge) {
            if (containedgraph != null) {
                makeCircleNode(to);
                containedgraph.print("\t");
                outputString(from);
                containedgraph.print(" -> ");
                outputString(to);
                labelEdge(edge);
                containedgraph.println(";");
            }
        }

        public static void addEdge(String from, String to) {
            addEdge(from, to, null);
        }

        public static void addEdge(String from, String to, String edge) {
            if (containedgraph != null) {
                containedgraph.print("\t");
                outputString(from);
                containedgraph.print(" -> ");
                outputString(to);
                labelEdge(edge);
                containedgraph.println(";");
            }
        }

        public static void closeGraph() {
            containedgraph.println("}");
            containedgraph.close();
            containedgraph = null;
        }
    }

    /**
     * Use the dot helper class to output this cfg as a Graph.
     */
    public void visitCFG (ControlFlowGraph cfg) {
        try {
            String filename = createMethodName(cfg.getMethod());
	    dot.openGraph(filename);
            cfg.visitBasicBlocks(new BasicBlockVisitor() {
                public void visitBasicBlock(BasicBlock bb) {
                    if (bb.isEntry()) {
                        if (bb.getNumberOfSuccessors() != 1)
                            throw new Error("entry bb has != 1 successors " + bb.getNumberOfSuccessors());
                        dot.addEntryEdge(bb.toString(), bb.getSuccessors().iterator().next().toString(), null);
                    } else
                    if (!bb.isExit()) {
                        ListIterator.Quad qit = bb.iterator();
                        StringBuffer l = new StringBuffer(" " + bb.toString() + "\\l");
                        HashSet allExceptions = new HashSet();
                        while (qit.hasNext()) {
                            l.append(" ");
                            Quad quad = qit.nextQuad();
                            l.append(dot.escape(quad.toString()));
                            l.append("\\l");
                            ListIterator.jq_Class exceptions = quad.getThrownExceptions().classIterator();
                            while (exceptions.hasNext()) {
                                allExceptions.add(exceptions.nextClass());
                            }
                        }
                        dot.userDefined("\t" + bb.toString() + " [shape=box,label=\"" + l + "\"];\n");

                        ListIterator.BasicBlock bit = bb.getSuccessors().basicBlockIterator();
                        while (bit.hasNext()) {
                            BasicBlock nextbb = bit.nextBasicBlock();
                            if (nextbb.isExit()) {
                                dot.addLeavingEdge(bb.toString(), nextbb.toString(), null);
                            } else {
                                dot.addEdge(bb.toString(), nextbb.toString());
                            }
                        }

                        Iterator eit = allExceptions.iterator();
                        while (eit.hasNext()) {
                            jq_Class exc = (jq_Class)eit.next();
                            ListIterator.ExceptionHandler mayCatch;
                            mayCatch = bb.getExceptionHandlers().mayCatch(exc).exceptionHandlerIterator();
                            while (mayCatch.hasNext()) {
                                ExceptionHandler exceptionHandler = mayCatch.nextExceptionHandler();
                                BasicBlock nextbb = exceptionHandler.getEntry();
                                dot.addEdge(bb.toString(), nextbb.toString(), exceptionHandler.getExceptionType().toString());
                            }
                            // if (bb.getExceptionHandlers().mustCatch(exc) == null) { }
                        }
                    }
                }
            });
        } finally {
            dot.closeGraph();
        }
    }
}
