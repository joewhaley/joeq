// ScanObject.java, created Tue Dec 10 14:02:22 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Memory.Manager;

import java.lang.reflect.Array;

import Allocator.DefaultHeapAllocator;
import Allocator.ObjectLayout;
import Clazz.jq_Array;
import Clazz.jq_Class;
import Clazz.jq_Reference;
import Clazz.jq_Type;
import Memory.HeapAddress;
import Run_Time.Debug;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class ScanObject {

    /**
     * Scans an object or array for internal object references and
     * processes those references (calls processPtrField)
     *
     * @param objRef  reference for object to be scanned (as int)
     */
    static void scanObjectOrArray(HeapAddress objRef) {

        Object obj = objRef.asObject();
        jq_Reference type = jq_Reference.getTypeOf(obj);
        if (type.isClassType()) {
            int[] referenceOffsets = ((jq_Class)type).getReferenceOffsets();
            for (int i = 0, n = referenceOffsets.length; i < n; i++) {
                DefaultHeapAllocator.processPtrField(objRef.offset(referenceOffsets[i]));
            }
        } else {
            jq_Type elementType = ((jq_Array)type).getElementType();
            if (elementType.isReferenceType()) {
                int num_elements = Array.getLength(obj);
                int numBytes = num_elements * HeapAddress.size();
                HeapAddress location = (HeapAddress) objRef.offset(ObjectLayout.ARRAY_ELEMENT_OFFSET);
                HeapAddress end = (HeapAddress) location.offset(numBytes);
                while (location.difference(end) < 0) {
                    DefaultHeapAllocator.processPtrField(location);
                    location =
                        (HeapAddress) location.offset(HeapAddress.size());
                }
            }
        }
    }

    static void scanObjectOrArray(Object objRef) {
        scanObjectOrArray(HeapAddress.addressOf(objRef));
    }

    public static boolean validateRefs(HeapAddress ref, int depth) {

        jq_Type type;

        if (ref.isNull())
            return true; // null is always valid

        // First check passed ref, before looking into it for refs
        if (!GCUtil.validRef(ref)) {
            Debug.write("ScanObject.validateRefs: Bad Ref = ");
            GCUtil.dumpRef(ref);
            //VM_Memory.dumpMemory(ref, 32, 32);
            // dump 16 words on either side of bad ref
            return false;
        }

        if (depth == 0)
            return true; //this ref valid, stop depth first scan

        type = jq_Reference.getTypeOf(ref.asObject());
        if (type.isClassType()) {
            int[] referenceOffsets = ((jq_Class)type).getReferenceOffsets();
            for (int i = 0, n = referenceOffsets.length; i < n; i++) {
                HeapAddress iref =
                     (HeapAddress) ref.offset(referenceOffsets[i]).peek();
                if (!validateRefs(iref, depth - 1)) {
                    Debug.write("Referenced from Object: Ref = ");
                    GCUtil.dumpRef(ref);
                    Debug.write("                  At Offset = ");
                    Debug.write(referenceOffsets[i]);
                    Debug.write("\n");
                    return false;
                }
            }
        } else {
            jq_Type elementType = ((jq_Array)type).getElementType();
            if (elementType.isReferenceType()) {
                int num_elements = Array.getLength(ref.asObject());
                int location = 0; // for arrays = offset of [0] entry
                int end = num_elements * HeapAddress.size();
                for (location = 0; location < end; location += HeapAddress.size()) {
                    HeapAddress iref = (HeapAddress) ref.offset(location).peek();
                    if (!validateRefs(iref, depth - 1)) {
                        Debug.write("Referenced from Array: Ref = ");
                        GCUtil.dumpRef(ref);
                        Debug.write("                  At Index = ");
                        Debug.write(location >> 2);
                        Debug.write("              Array Length = ");
                        Debug.write(num_elements);
                        Debug.write("\n");
                        return false;
                    }
                }
            }
        }
        return true;
    } // validateRefs

}
