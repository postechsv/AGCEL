# Semantics for Promela as our modeling language

### Notes
* why flattening may not work
>> consider if ... :: do ... od :: ... fi. In this case, the structure of do ... od must be preserved.
>> otherwise, when do branch is taken and reached the end, we don't know where to go back.
