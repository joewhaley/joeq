/**
 * BitArray
 *
 * Created on Sep 24, 2002, 7:58:38 PM
 *
 * @author laudney <bin_ren@myrealbox.com>
 * @version 0.1
 */
package Util;

public class BitArray {
    private int[] bits;

    public BitArray(int size_in_bits) {
        bits = new int[size_in_bits/32];
    }

    public void set(int index) {
        bits[index/32] |= (1 << (index%32));
    }

    public void unset(int index) {
        bits[index/32] &= ~(1 << (index%32));
    }

    public boolean isSet(int index) {
        return (bits[index/32] & (1 << (index%32))) != 0;
    }

    public class BitArrayIterator implements java.util.Iterator {
        private int currentIndex = 0;

        public int nextIndex() {
            if (currentIndex == bits.length*32-1) {
                throw new java.util.NoSuchElementException();
            } else {
                return currentIndex + 1;
            }
        }

        public Object next() {
            int nextIndex = nextIndex();
            ++currentIndex;
            if (isSet(nextIndex)) {
                return new Integer(1);
            } else {
                return new Integer(0);
            }
        }

        public boolean hasNext() {
            return currentIndex != bits.length*32-1;
        }

        public void remove() { throw new UnsupportedOperationException(); }

    }
}
