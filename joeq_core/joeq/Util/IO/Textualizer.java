// Textualizer.java, created Oct 26, 2003 5:15:38 PM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Util.IO;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.StringTokenizer;

import Util.Assert;
import Util.Collections.IndexMap;
import Util.Collections.IndexedMap;
import Util.Collections.Pair;

/**
 * Textualizer
 * 
 * @author John Whaley
 * @version $Id$
 */
public interface Textualizer {

    Textualizable readObject() throws IOException;
    Textualizable readReference() throws IOException;
    
    void writeTypeOf(Textualizable object) throws IOException;
    void writeObject(Textualizable object) throws IOException;
    void writeEdge(String edgeName, Textualizable object) throws IOException;
    void writeReference(Textualizable object) throws IOException;

    void writeBytes(String s) throws IOException;
    
    StringTokenizer nextLine() throws IOException;

    int getIndex(Textualizable object);
    boolean contains(Textualizable object);
    
    public static class Simple implements Textualizer {
        protected DataInput in;
        protected DataOutput out;
        protected StringTokenizer st;
        
        public Simple(DataInput in) {
            this.in = in;
        }
        
        public Simple(DataOutput out) {
            this.out = out;
        }
        
        public StringTokenizer nextLine() throws IOException {
            return st = new StringTokenizer(in.readLine());
        }
        
        protected void updateTokenizer() throws IOException {
            if (st == null || !st.hasMoreElements())
                st = new StringTokenizer(in.readLine());
        }
        
        public Textualizable readObject() throws IOException {
            updateTokenizer();
            String className = st.nextToken();
            if (className.equals("null")) return null;
            try {
                Class c = Class.forName(className);
                Method m = c.getMethod("read", new Class[] {StringTokenizer.class});
                return (Textualizable) m.invoke(null, new Object[] {st});
            } catch (ClassNotFoundException e) {
                e.printStackTrace();
            } catch (SecurityException e) {
                e.printStackTrace();
            } catch (NoSuchMethodException e) {
                e.printStackTrace();
            } catch (IllegalArgumentException e) {
                e.printStackTrace();
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            } catch (InvocationTargetException e) {
                e.printStackTrace();
            }
            return null;
        }
        
        public Textualizable readReference() throws IOException {
            return readObject();
        }
        
        public void writeTypeOf(Textualizable object) throws IOException {
            if (object != null) {
                out.writeBytes(object.getClass().getName());
                out.write(' ');
            }
        }
        
        public void writeObject(Textualizable object) throws IOException {
            if (object != null) {
                object.write(this);
            } else {
                out.writeBytes("null");
            }
        }
        
        public void writeReference(Textualizable object) throws IOException {
            writeObject(object);
        }
        
        public void writeEdge(String edgeName, Textualizable object) throws IOException {
            throw new InternalError();
        }

        public void writeBytes(String s) throws IOException {
            out.writeBytes(s);
        }

        public int getIndex(Textualizable object) {
            throw new InternalError();
        }
        
        public boolean contains(Textualizable object) {
            throw new InternalError();
        }
    }
    
    public static class Map extends Simple {
        protected IndexedMap map;
        protected java.util.Map deferredEdges;
        
        public Map(DataInput in) {
            this(in, new IndexMap(""));
        }
        
        public Map(DataInput in, IndexedMap map) {
            super(in);
            this.map = map;
        }
        
        public Map(DataOutput out, IndexedMap map) {
            super(out);
            this.map = map;
        }
        
        public Textualizable readObject() throws IOException {
            Textualizable t = super.readObject();
            int s = map.size();
            int f = map.get(t);
            Assert._assert(f == s);
            if (false) readEdges(t);
            if (deferredEdges != null) {
                Collection d = (Collection) deferredEdges.get(new Integer(f));
                if (d != null) {
                    for (Iterator i = d.iterator(); i.hasNext(); ) {
                        Pair def = (Pair) i.next();
                        Textualizable source = (Textualizable) def.left;
                        String edgeName = (String) def.right;
                        source.addEdge(edgeName, t);
                    }
                    deferredEdges.remove(t);
                }
            }
            return t;
        }
        
        public void readEdges(Textualizable t) {
            while (st.hasMoreTokens()) {
                String edgeName = st.nextToken();
                int index = Integer.parseInt(st.nextToken());
                if (index >= map.size()) {
                    Integer i = new Integer(index);
                    Collection c = (Collection) deferredEdges.get(i);
                    if (c == null) deferredEdges.put(i, c = new LinkedList());
                    c.add(new Pair(t, edgeName));
                } else {
                    Textualizable t2 = (Textualizable) map.get(index);
                    t.addEdge(edgeName, t2);
                }
            }
        }
        
        public Textualizable readReference() throws IOException {
            updateTokenizer();
            int id = Integer.parseInt(st.nextToken());
            return (Textualizable) map.get(id);
        }
        
        public void writeObject(Textualizable object) throws IOException {
            //map.get(object);
            super.writeObject(object);
            if (object != null) object.writeEdges(this);
        }
        
        public void writeReference(Textualizable object) throws IOException {
            if (!map.contains(object)) {
                System.out.println("Not in map: "+object);
                writeObject(object);
            } else {
                int id = map.get(object);
                out.writeBytes(Integer.toString(id));
            }
        }

        public void writeEdge(String edgeName, Textualizable target) throws IOException {
            out.writeByte(' ');
            out.writeBytes(edgeName);
            out.writeByte(' ');
            map.get(target);
            writeReference(target);
        }
        
        public int getIndex(Textualizable object) {
            Assert._assert(map.contains(object));
            return map.get(object);
        }
        
        public boolean contains(Textualizable object) {
            return map.contains(object);
        }
        
        public IndexedMap getMap() {
            return map;
        }
    }
    
}
