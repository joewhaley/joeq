// UnionFind.java, created Mar 25, 2004 8:11:25 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.Collections;

/**
 * Union-Find data structure.  Works with objects or int's.
 * 
 * @author jwhaley
 * @version $Id$
 */
public class UnionFind {
    
    private int[] array;
    private IndexMap map;
    
    public UnionFind(int numElements) {
        array = new int[numElements];
        for (int i = 0; i < array.length; i++)
            array[i] = -1;
        map = new IndexMap("UnionFindMap");
    }
    
    public int add(Object x) {
        return map.get(x);
    }
    
    public void union(Object root1, Object root2) {
        union(map.get(root1), map.get(root2));
    }
    
    /**
     * Union two disjoint sets using the height heuristic. For simplicity, we
     * assume root1 and root2 are distinct and represent set names.
     * 
     * @param root1
     *                the root of set 1.
     * @param root2
     *                the root of set 2.
     */
    public void union(int root1, int root2) {
        if (array[root2] < array[root1]) /* root2 is deeper */
            array[root1] = root2; /* Make root2 new root */
        else {
            if (array[root1] == array[root2])
                array[root1]--; /* Update height if same */
            array[root2] = root1; /* Make root1 new root */
        }
    }
    
    public Object find(Object x) {
        return map.get(find(map.get(x)));
    }
    
    /**
     * Perform a find with path compression. Error checks omitted again for
     * simplicity.
     * 
     * @param x
     *                the element being searched for.
     * @return the set containing x.
     */
    public int find(int x) {
        if (array[x] < 0)
            return x;
        else
            return array[x] = find(array[x]);
    }
}
