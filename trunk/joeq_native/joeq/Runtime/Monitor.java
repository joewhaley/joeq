/*
 * Monitor.java
 *
 * Created on January 16, 2001, 9:58 PM
 *
 */

package Run_Time;

import Allocator.DefaultHeapAllocator;
import Allocator.ObjectLayout;
import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Clazz.jq_StaticMethod;
import Main.jq;
import Memory.HeapAddress;
import Scheduler.jq_Thread;
import Util.Strings;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class Monitor implements ObjectLayout {

    public static /*final*/ boolean TRACE = false;
    
    private Monitor() {}
    
    int atomic_count = 0;  // -1 means no threads; 0 means one thread (monitor_owner)
    jq_Thread monitor_owner;
    int entry_count = 0;    // 1 means locked once.
    int/*CPointer*/ semaphore;
    
    /** Returns the depth of the lock on the given object. */
    public static int getLockEntryCount(Object k) {
        int lockword = HeapAddress.addressOf(k).offset(STATUS_WORD_OFFSET).peek4();
        if (lockword < 0) {
            Monitor m = getMonitor(lockword);
            if (TRACE) SystemInterface.debugmsg("Getting fat lock entry count: "+m.entry_count);
            return m.entry_count;
        }
        int c = ((lockword & LOCK_COUNT_MASK) >> LOCK_COUNT_SHIFT);
        if ((lockword & THREAD_ID_MASK) != 0) ++c;
        if (TRACE) SystemInterface.debugmsg("Getting thin lock entry count, lockword="+Strings.hex8(lockword)+", count="+c);
        return c;
    }
    
    /** Monitorenter runtime routine.
     *  Checks for thin lock usage, otherwise falls back to inflated locks.
     */
    public static void monitorenter(Object k) {
        jq_Thread t = Unsafe.getThreadBlock();
        int tid = t.getThreadId(); // pre-shifted thread id
        
        // attempt fast path: object is not locked.
        HeapAddress status_address = (HeapAddress) HeapAddress.addressOf(k).offset(STATUS_WORD_OFFSET);
        int status_flags = status_address.peek4() & STATUS_FLAGS_MASK;
        int newlockword = status_flags | tid;
        int oldlockword = status_address.atomicCas4(status_flags, newlockword);
        if (Unsafe.isEQ()) {
            // fast path: not locked
            return;
        }
        
        // object is locked or has an inflated lock.
        int counter = oldlockword ^ newlockword; // if tid's are equal, this extracts the counter.
        if (counter >= LOCK_COUNT_MASK) {
            // slow path: other thread owns thin lock, or entry counter == max
            int entrycount;
            if (counter == LOCK_COUNT_MASK) {
                // thin lock entry counter == max, so we need to inflate ourselves.
                if (TRACE) SystemInterface.debugmsg("Thin lock counter overflow, inflating lock...");
                entrycount = (LOCK_COUNT_MASK >> LOCK_COUNT_SHIFT)+2;
                Monitor m = allocateInflatedLock();
                m.monitor_owner = t;
                m.entry_count = entrycount;
                newlockword = HeapAddress.addressOf(m).to32BitValue() | LOCK_EXPANDED | status_flags;
                // we own the lock, so a simple write is sufficient.
                jq.Assert(HeapAddress.addressOf(k).offset(STATUS_WORD_OFFSET).peek4() == oldlockword);
                HeapAddress.addressOf(k).offset(STATUS_WORD_OFFSET).poke4(newlockword);
            } else {
                // thin lock owned by another thread.
                if (TRACE) SystemInterface.debugmsg(t+" tid "+Strings.hex(tid)+": Lock contention with tid "+Strings.hex(oldlockword & THREAD_ID_MASK)+", inflating...");
                entrycount = 1;
                Monitor m = allocateInflatedLock();
                m.monitor_owner = t;
                m.entry_count = entrycount;
                // install an inflated lock.
                installInflatedLock(k, m);
            }
        } else if (counter < 0) {
            // slow path 2: high bit of lock word is set --> inflated lock
            Monitor m = getMonitor(oldlockword);
            m.lock(t);
        } else {
            // not-quite-so-fast path: locked by current thread.  increment counter.
            // we own the lock, so a simple write is sufficient.
            HeapAddress.addressOf(k).offset(STATUS_WORD_OFFSET).poke4(oldlockword+LOCK_COUNT_INC);
        }
    }
    
    /** Monitorexit runtime routine.
     *  Checks for thin lock usage, otherwise falls back to inflated locks.
     */
    public static void monitorexit(Object k) {
        jq_Thread t = Unsafe.getThreadBlock();
        int tid = t.getThreadId(); // pre-shifted
        HeapAddress status_address = (HeapAddress) HeapAddress.addressOf(k).offset(STATUS_WORD_OFFSET);
        int oldlockword = status_address.peek4();
        // if not inflated and tid matches, this contains status flags and counter
        int counter = oldlockword ^ tid;
        if (counter < 0) {
            // inflated lock
            Monitor m = getMonitor(oldlockword);
            m.unlock(t);
        } else if (counter <= STATUS_FLAGS_MASK) {
            // owned by us and count is zero.  clear tid field.
            status_address.atomicAnd(STATUS_FLAGS_MASK);
        } else if (counter <= (LOCK_COUNT_MASK | STATUS_FLAGS_MASK)) {
            // owned by us but count is non-zero.  decrement count.
            status_address.atomicSub(LOCK_COUNT_INC);
        } else {
            // lock not owned by us!
            //if (TRACE)
                SystemInterface.debugmsg("Thin lock not owned by us ("+Strings.hex8(tid)+")! lockword="+Strings.hex8(oldlockword));
            throw new IllegalMonitorStateException();
        }
    }
    
    /** Get the Monitor object associated with this lockword. */
    public static Monitor getMonitor(int lockword) {
        int word = lockword & (~LOCK_EXPANDED & ~STATUS_FLAGS_MASK);
        HeapAddress a = HeapAddress.address32(word);
        return (Monitor)a.asObject();
    }
    
    public static Monitor allocateInflatedLock() {
        // Monitor must be 8-byte aligned.
        return (Monitor)DefaultHeapAllocator.allocateObjectAlign8(_class.getInstanceSize(), _class.getVTable());
    }
    
    public void free() {
        // nop. we will be garbage collected.
    }
    
    /** Installs an inflated lock on the given object.
     *  Uses a spin-loop to wait until the object is unlocked or inflated.
     */
    public static void installInflatedLock(Object k, Monitor m) {
        jq.Assert(m.monitor_owner == Unsafe.getThreadBlock());
        jq.Assert(m.entry_count >= 1);
        for (;;) {
            HeapAddress status_address = (HeapAddress) HeapAddress.addressOf(k).offset(STATUS_WORD_OFFSET);
            int oldlockword = status_address.peek4();
            if (oldlockword < 0) {
                // inflated by another thread!  free our inflated lock and use that one.
                jq.Assert(m.entry_count == 1);
                m.free();
                Monitor m2 = getMonitor(oldlockword);
                if (TRACE) SystemInterface.debugmsg("Inflated by another thread! lockword="+Strings.hex8(oldlockword)+" lock="+m2);
                jq.Assert(m != m2);
                m2.lock(Unsafe.getThreadBlock());
                return;
            }
            int status_flags = oldlockword & STATUS_FLAGS_MASK;
            HeapAddress m_addr = HeapAddress.addressOf(m);
            if ((m_addr.to32BitValue() & STATUS_FLAGS_MASK) != 0 ||
                (m_addr.to32BitValue() & LOCK_EXPANDED) != 0) {
                jq.UNREACHABLE("Monitor object has address "+m_addr.stringRep());
            }
            int newlockword = m_addr.to32BitValue() | LOCK_EXPANDED | status_flags;
            status_address.atomicCas4(status_flags, newlockword);
            if (Unsafe.isEQ()) {
                // successfully obtained inflated lock.
                if (TRACE) SystemInterface.debugmsg("Obtained inflated lock! new lockword="+Strings.hex8(newlockword));
                return;
            } else {
                if (TRACE) SystemInterface.debugmsg("Failed to obtain inflated lock, lockword was "+Strings.hex8(oldlockword));
            }
            // another thread has a thin lock on this object.  yield to scheduler.
            Thread.yield();
        }
    }

    /** Lock this monitor with the given thread block.
     */
    public void lock(jq_Thread t) {
        jq_Thread m_t = this.monitor_owner;
        if (m_t == t) {
            // we own the lock.
            jq.Assert(this.atomic_count >= 0);
            jq.Assert(this.entry_count > 0);
            ++this.entry_count;
            if (TRACE) SystemInterface.debugmsg("We ("+t+") own lock "+this+", incrementing entry count: "+this.entry_count);
            return;
        }
        if (TRACE) SystemInterface.debugmsg("We ("+t+") are attempting to obtain lock "+this);
        // another thread or no thread owns the lock. increase atomic count.
        HeapAddress ac_loc = (HeapAddress) HeapAddress.addressOf(this).offset(_atomic_count.getOffset());
        ac_loc.atomicAdd(1);
        if (!Unsafe.isEQ()) {
            // someone else already owns the lock.
            if (TRACE) SystemInterface.debugmsg("Lock "+this+" cannot be obtained (owned by "+m_t+", or there are other waiters); waiting on semaphore ("+this.atomic_count+" waiters)");
            // create a semaphore if there isn't one already, and wait on it.
            this.waitOnSemaphore();
            if (TRACE) SystemInterface.debugmsg("We ("+t+") finished waiting on "+this);
        } else {
            if (TRACE) SystemInterface.debugmsg(this+" is unlocked, we ("+t+") obtain it.");
        }
        jq.Assert(this.monitor_owner == null);
        jq.Assert(this.entry_count == 0);
        jq.Assert(this.atomic_count >= 0);
        if (TRACE) SystemInterface.debugmsg("We ("+t+") obtained lock "+this);
        this.monitor_owner = t;
        this.entry_count = 1;
    }
    
    /** Unlock this monitor with the given thread block.
     */
    public void unlock(jq_Thread t) {
        jq_Thread m_t = this.monitor_owner;
        if (m_t != t) {
            // lock not owned by us!
            //if (TRACE)
                SystemInterface.debugmsg("We ("+t+") tried to unlock lock "+this+" owned by "+m_t);
            throw new IllegalMonitorStateException();
        }
        if (--this.entry_count > 0) {
            // not zero yet.
            if (TRACE) SystemInterface.debugmsg("Decrementing lock "+this+" entry count "+this.entry_count);
            return;
        }
        if (TRACE) SystemInterface.debugmsg("We ("+t+") are unlocking lock "+this+", current waiters="+this.atomic_count);
        this.monitor_owner = null;
        HeapAddress ac_loc = (HeapAddress) HeapAddress.addressOf(this).offset(_atomic_count.getOffset());
        ac_loc.atomicSub(1);
        if (Unsafe.isGE()) {
            // threads are waiting on us, release the semaphore.
            if (TRACE) SystemInterface.debugmsg((this.atomic_count+1)+" threads are waiting on released lock "+this+", releasing semaphore.");
            this.releaseSemaphore();
        } else {
            if (TRACE) SystemInterface.debugmsg("No threads are waiting on released lock "+this+".");
        }
    }
    
    /** Create a semaphore if there isn't one already, and wait on it.
     */
    public void waitOnSemaphore() {
        if (this.semaphore == 0) {
            this.semaphore = SystemInterface.init_semaphore();
        }
        // TODO: integrate waiting on a semaphore into the scheduler
        for (;;) {
            int rc = SystemInterface.wait_for_single_object(this.semaphore, 10); // timeout
            if (rc == SystemInterface.WAIT_TIMEOUT) {
                Thread.yield();
                continue;
            } else if (rc == 0) {
                return;
            } else {
                SystemInterface.debugmsg("Bad return value from WaitForSingleObject: "+rc);
            }
        }
    }
    /** Create a semaphore if there isn't one already, and release it.
     */
    public void releaseSemaphore() {
        if (this.semaphore == 0) {
            this.semaphore = SystemInterface.init_semaphore();
        }
        int rc = SystemInterface.release_semaphore(this.semaphore, 1);
    }
    
    public static final jq_Class _class;
    public static final jq_StaticMethod _monitorenter;
    public static final jq_StaticMethod _monitorexit;
    public static final jq_InstanceField _atomic_count;
    static {
        _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("LRun_Time/Monitor;");
        _monitorenter = _class.getOrCreateStaticMethod("monitorenter", "(Ljava/lang/Object;)V");
        _monitorexit = _class.getOrCreateStaticMethod("monitorexit", "(Ljava/lang/Object;)V");
        _atomic_count = _class.getOrCreateInstanceField("atomic_count", "I");
    }

}
