'''
Keys and Rooms
Medium

There are N rooms and you start in room 0.  Each room has a distinct number in 0, 1, 2, ..., N-1, and each room may have some keys to access the next room. 

Formally, each room i has a list of keys rooms[i], and each key rooms[i][j] is an integer in [0, 1, ..., N-1] where N = rooms.length.  A key rooms[i][j] = v opens the room with number v.

Initially, all the rooms start locked (except for room 0). 

You can walk back and forth between rooms freely.

Return true if and only if you can enter every room.
'''

class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        seen = [False for r in rooms]
        seen[0] = True
        
        keys = []
        keys.append(0)
        
        while len(keys) != 0:
            cur = keys.pop()
            for new_key in rooms[cur]:
                if not seen[new_key]:
                    seen[new_key] = True
                    keys.append(new_key)
                    
        if False in seen:
            return False
        return True