/*
 * SetRepository
 *
 * Created on September 20, 2002, 6:14 AM
 *
 */

package Util;

import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

import Main.jq;

/**
 * @author  John Whaley
 * @version $Id$
 */
public class SetRepository extends SetFactory {

    public static final boolean USE_HASHCODES = true;
    public static final boolean USE_SIZES     = false;
    
    public static final boolean VerifyAssertions = true;

    public static class LinkedHashSetFactory extends SetFactory {
        private LinkedHashSetFactory() { }
        public static final LinkedHashSetFactory INSTANCE = new LinkedHashSetFactory();
        
        /** Generates a new mutable <code>Set</code>, using the elements
            of <code>c</code> as a template for its initial contents. 
        */ 
        public final Set makeSet(Collection c) {
            LinkedHashSet set = new LinkedHashSet();
            set.addAll(c);
            return set;
        }
    }
    public static class SimpleHashSetFactory extends MapFactory {
        private SimpleHashSetFactory() { }
        public static final SimpleHashSetFactory INSTANCE = new SimpleHashSetFactory();
        
        /** Generates a new <code>Map</code>, using the entries of
            <code>map</code> as a template for its initial mappings. 
        */
        public final Map makeMap(Map map) {
            SimpleHashSet set = new SimpleHashSet();
            set.putAll(map);
            return set;
        }
    }

    Map cache;
    SetFactory setFactory;
    MapFactory entryFactory;
    
    public SetRepository() {
        if (USE_HASHCODES) {
            cache = new HashMap();
            setFactory = LinkedHashSetFactory.INSTANCE;
            entryFactory = LightMap.Factory.INSTANCE;
        } else if (USE_SIZES) {
            cache = new LightMap();
            setFactory = LinkedHashSetFactory.INSTANCE;
            entryFactory = SimpleHashSetFactory.INSTANCE;
        } else {
            jq.UNREACHABLE();
        }
    }
    
    public final Set makeSet(Collection c) {
        return SharedSet.make(this, c);
    }
    
    public SharedSet getUnion(Collection sets, boolean disjoint) {
        Object setIdentifier;
        if (disjoint)
            setIdentifier = calculateSetIdentifier_disjoint(sets);
        else
            setIdentifier = calculateSetIdentifier(sets);
        Map entry = (Map) cache.get(setIdentifier);
        if (entry == null) {
            cache.put(setIdentifier, entry = entryFactory.makeMap());
        }
        if (USE_HASHCODES) {
            SharedSet resultSet = SharedSet.makeUnion(this, sets);
            SharedSet resultSet2 = (SharedSet) entry.get(resultSet);
            if (resultSet2 != null) return resultSet2;
            entry.put(resultSet, resultSet);
            return resultSet;
        } else if (USE_SIZES) {
            int newHashCode;
            if (disjoint)
                newHashCode = calculateHashcode_disjoint(sets);
            else
                newHashCode = calculateHashcode(sets);
            SimpleHashSet s = (SimpleHashSet) entry;
            Iterator i = s.getMatchingHashcode(newHashCode).iterator();
uphere:
            while (i.hasNext()) {
                SharedSet hs = (SharedSet) i.next();
                Iterator j = sets.iterator();
                while (j.hasNext()) {
                    Set s1 = (Set) j.next();
                    if (!hs.containsAll(s1))
                        continue uphere;
                }
                return hs;
            }
            SharedSet resultSet = SharedSet.makeUnion(this, sets);
            s.put(resultSet, resultSet);
            return resultSet;
        } else {
            jq.UNREACHABLE(); return null;
        }
    }
    
    static boolean checkDisjoint(Collection sets) {
        Iterator i = sets.iterator();
        if (!i.hasNext()) return true;
        Set s1 = (Set) i.next();
        while (i.hasNext()) {
            s1 = (Set) i.next();
            for (Iterator j = s1.iterator(); j.hasNext(); ) {
                Object o = j.next();
                for (Iterator k = sets.iterator(); ; ) {
                    Set s2 = (Set) k.next();
                    if (s2 == s1) break;
                    if (s2.contains(o)) return false;
                }
            }
        }
        return true;
    }
    
    static int calculateHashcode_disjoint(Collection sets) {
        int newHashCode = 0;
        for (Iterator i = sets.iterator(); i.hasNext(); ) {
            Set s = (Set) i.next();
            newHashCode += s.hashCode();
        }
        if (VerifyAssertions) jq.Assert(checkDisjoint(sets) == true);
        return newHashCode;
    }
    
    static int calculateSize_disjoint(Collection sets) {
        int newSize = 0;
        for (Iterator i = sets.iterator(); i.hasNext(); ) {
            Set s = (Set) i.next();
            newSize += s.size();
        }
        if (VerifyAssertions) jq.Assert(checkDisjoint(sets) == true);
        return newSize;
    }
    
    static int calculateHashcode(Collection sets) {
        int newHashCode = 0;
        Iterator i = sets.iterator();
        if (!i.hasNext()) return newHashCode;
        Set s1 = (Set) i.next();
        newHashCode = s1.hashCode();
        while (i.hasNext()) {
            s1 = (Set) i.next();
uphere:
            for (Iterator j = s1.iterator(); j.hasNext(); ) {
                Object o = j.next();
                for (Iterator k = sets.iterator(); ; ) {
                    Set s2 = (Set) k.next();
                    if (s2 == s1) break;
                    if (s2.contains(o)) break uphere;
                }
                newHashCode += o.hashCode();
            }
        }
        return newHashCode;
    }
    
    static int calculateSize(Collection sets) {
        int newSize = 0;
        Iterator i = sets.iterator();
        if (!i.hasNext()) return newSize;
        Set s1 = (Set) i.next();
        newSize = s1.size();
        while (i.hasNext()) {
            s1 = (Set) i.next();
uphere:
            for (Iterator j = s1.iterator(); j.hasNext(); ) {
                Object o = j.next();
                for (Iterator k = sets.iterator(); ; ) {
                    Set s2 = (Set) k.next();
                    if (s2 == s1) break;
                    if (s2.contains(o)) break uphere;
                }
                ++newSize;
            }
        }
        return newSize;
    }
    
    public static Object calculateSetIdentifier_disjoint(Collection sets) {
        if (USE_HASHCODES) {
            int newHashCode = calculateHashcode_disjoint(sets);
            return new Integer(newHashCode);
        } else if (USE_SIZES) {
            int newSize = calculateSize_disjoint(sets);
            return new Integer(newSize);
        } else {
            jq.UNREACHABLE();
            return null;
        }
    }
    
    public static Object calculateSetIdentifier(Collection sets) {
        if (USE_HASHCODES) {
            int newHashCode = calculateHashcode(sets);
            return new Integer(newHashCode);
        } else if (USE_SIZES) {
            int newSize = calculateSize(sets);
            return new Integer(newSize);
        } else {
            jq.UNREACHABLE();
            return null;
        }
    }
    
    public static class SharedSet implements Set {
        private final Set set;
        private final SetRepository repository;
        
        public static SharedSet make(SetRepository repository, Collection s) {
            return new SharedSet(repository, s);
        }
        public static SharedSet makeUnion(SetRepository repository, Collection sets) {
            Iterator i = sets.iterator();
            Set s = (Set) i.next();
            SharedSet that = new SharedSet(repository, s);
            while (i.hasNext()) {
                s = (Set) i.next();
                that.set.addAll(s);
            }
            return that;
        }
        private SharedSet(SetRepository repository) {
            this.repository = repository;
            this.set = repository.setFactory.makeSet();
        }
        private SharedSet(SetRepository repository, Collection s) {
            this.repository = repository;
            this.set = repository.setFactory.makeSet(s);
        }
        
        public SharedSet copyAndAddAll(Set s, boolean disjoint) {
            return repository.getUnion(Default.pair(this.set, s), disjoint);
        }
        
        public SharedSet copyAndAddAllSets(Collection sets, boolean disjoint) {
            return repository.getUnion(sets, disjoint);
        }
        
        /**
         * @see java.util.Collection#add(Object)
         */
        public boolean add(Object arg0) {
            throw new UnsupportedOperationException();
        }

        /**
         * @see java.util.Collection#addAll(Collection)
         */
        public boolean addAll(Collection arg0) {
            throw new UnsupportedOperationException();
        }

        /**
         * @see java.util.Collection#clear()
         */
        public void clear() {
            throw new UnsupportedOperationException();
        }

        /**
         * @see java.util.Collection#contains(Object)
         */
        public boolean contains(Object arg0) {
            return set.contains(arg0);
        }

        /**
         * @see java.util.Collection#containsAll(Collection)
         */
        public boolean containsAll(Collection arg0) {
            return set.containsAll(arg0);
        }

        /**
         * @see java.util.Collection#isEmpty()
         */
        public boolean isEmpty() {
            return set.isEmpty();
        }

        /**
         * @see java.util.Collection#iterator()
         */
        public Iterator iterator() {
            return set.iterator();
        }

        /**
         * @see java.util.Collection#remove(Object)
         */
        public boolean remove(Object arg0) {
            throw new UnsupportedOperationException();
        }

        /**
         * @see java.util.Collection#removeAll(Collection)
         */
        public boolean removeAll(Collection arg0) {
            throw new UnsupportedOperationException();
        }

        /**
         * @see java.util.Collection#retainAll(Collection)
         */
        public boolean retainAll(Collection arg0) {
            throw new UnsupportedOperationException();
        }

        /**
         * @see java.util.Collection#size()
         */
        public int size() {
            return set.size();
        }

        /**
         * @see java.util.Collection#toArray()
         */
        public Object[] toArray() {
            return set.toArray();
        }

        /**
         * @see java.util.Collection#toArray(Object[])
         */
        public Object[] toArray(Object[] arg0) {
            return set.toArray(arg0);
        }

    }
    
}
