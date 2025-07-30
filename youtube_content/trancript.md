what's up engineers indydev Dan here 
imagine opening your codebase and having 
your AI coding tool instantly understand 
your codebase better and faster than you 
can read the readme there are three 
simple folders that can unlock that 
superpower for you in every project you 
touch we're going to break them down 
into their atoms so you can use them to 
increase your compute advantage the AI 
coding tool you use does not change this 
one critical fact of engineering in the 
generative AI age doesn't matter if 
you're a cursor fan Windsor Fline Codeex 
or even Claude code you already know 
what this idea is context is everything 
context is king if your agent can't see 
critical information it simply cannot 
build what you need that's what these 
three essential directories solve 
comprehensively and systematically let's 
talk about these three essential 
directories and why they're valuable for 
your engineering 
work let's start with the foundation AI 
docs think of this as your AI coding 
tools persistent memory a knowledge 
repository your AI tools can instantly 
access inside of the cloud code is 
programmable codebase a big idea we 
discussed in last week's video you can 
see we have an AI docs directory inside 
of this directory we have two markdown 
files cloud code best practices and open 
AI's agent SDK i can now boot up any 
agent and have them quickly read these 
files they can then turn around to get 
work done quickly so what goes inside of 
AI docs here you have thirdparty API 
documentation integration details you 
have custom patterns and conventions any 
implementation notes anything specific 
to your codebase it all goes in AI docs 
i mostly use this for third-party 
documentation so that I can quickly ramp 
up my code bases over and over and over 
the AI docs directory is a persistent 
database for your AI agents 
so we of course have the specs directory 
what goes in the specs directory specs 
is short for specification which also 
just means plan you might know these as 
PRDs product documents whatever you want 
to call them these are the new units of 
getting massive amounts of work done 
with your AI coding tools and now with 
your agentic coding tools we can now use 
multiple tools inside of single prompt 
with powerful agentic coding tools like 
claw code cursor line so on and so forth 
the specs directory is the most 
important folder in your entire codebase 
this is where we write great plans this 
is where we scale up our compute and do 
more work than ever in single massive 
swings this 1,00 token prompt expanded 
into an entire codebase this is due to 
the fact that we are agentic coding 
right we can write self validating loops 
inside of our prompt remember agentic 
coding is a superset of AI coding a 
massive superset and great tools like 
Claw Code allow us to take all of our 
plans from all of our repositories from 
the specs directory and blow them out 
into full-on code bases and features 
this is why you should always have a 
specs directory that details the plan 
for all the work that you're going to 
hand off to your powerful agentic coding 
tools if you're still iteratively 
prompting back and forth and back and 
forth and back and forth I can guarantee 
you you are wasting time and you're not 
scaling your compute as much as you 
could be sit down take your time think 
plan and then build there's a powerful 
new planning technique that we're going 
to showcase in this video where you can 
use compute inside of your plan to 
iterate faster with your AI coding tool 
the key idea here is very simple every 
principled AI coding member knows this 
and everyone that's been following this 
channel knows this as well the plan is 
the prompt and great planning is great 
prompting you can scale up what you can 
do by writing a detailed comprehensive 
spec plan PRD whatever you want to call 
it you can then hand this work off to 
your AI coding tools and your agentic 
coding tool and they will get the work 
done for you 
so every codebase I build in now has the 
AI docs directory the specs directory 
for plans 
and claude now dotcloud is a new 
emerging directory to be super clear 
this is specific to claw code but what 
you write in these directories is not 
specific to claw code if we go to the 
just prompt codebase open up.claude and 
go into the commands directory you can 
see we have several different commands 
so what are these and how are they 
useful for scaling our engineering work 
these are nothing but prompts if we open 
up cloud code here in the just prompt 
codebase and we type slash you can see 
the names of all these commands right at 
the top here these are reusable runnable 
prompts that we can use across sessions 
the most important reusable prompt that 
I recommend you set up inside of all 
your codebases is the context priming 
prompt this is where you prompt cloud 
code codeex cursor whatever tool you're 
using this is not cloud code specific 
right the names of these directories can 
really be anything so if you were to 
prime or just prompt server here we'll 
do the basic context prime it's going to 
run through these commands right using 
tool calls it's going to read the readme 
then run get ls files to understand the 
context of this project so I recommend 
you set this up in every codebase so 
that you can quickly operate on the 
files and the ideas that matter what is 
the big idea of what we're doing here 
we're making it easy to set up new 
instances of our agentic tooling over 
time okay and by over time I mean on the 
day-to-day basis but also on a session 
to session basis if you've used cloud 
code or codeex or any one of your AI 
coding tools they will run out of 
context you can see the current context 
windows of the state-of-the-art models a 
lot of these are limited to 200K or 1 
million tokens when using your AI coding 
tools you'll eventually run out of 
context and then you'll have to reset so 
this is what the context prime does and 
this is what the do cloud commands 
directory gives you specifically for 
cloud code but you can deploy this 
across any AI coding tool right so if I 
were to open up a new window here and 
open 
codeex-m3 copy the relative path execute 
this this serves the exact same function 
okay and what are we doing here we're 
doing the same thing we did in cloud 
code we're setting up the initial 
context for the AI coding tool so that 
it understands everything so that knows 
where everything is okay you can see 
there it spatter out a nice summary and 
now it's ready to go right it context is 
prime these directories are not limited 
to context priming we built out a ultra 
diff review where we created a diff and 
then we had multiple language models 
review the diff and offer feedback this 
is something we're going to be talking 
about a lot on the channel the 
capabilities of your prompts are now 
unlimited thanks to Agentic coding tools 
you can run any tool you can run custom 
MCP servers like we have here you can do 
a tremendous amount by having reusable 
prompts inside your codebase so these 
are the three essential directories I 
have in every single one of my code 
bases now AI docs is the persistent 
knowledge base for your AI coding tools 
specs is where you define your plans 
it's where you define all the work you 
want done so that you can hand it off to 
your AI coding tool and to your agentic 
coding tools and then we have the 
dotcloud again you can name these 
whatever you want this is where you 
place your reusable prompts that you 
want at the ready inside of your agentic 
coding tool reusable prompts are an 
essential pattern even outside of a 
coding you want to be able to use 
compute over and over in different 
shapes and forms and you do that by 
creating prompts you can reuse validate 
and improve especially if you're 
building out 
[Music] 
emails what does this look like in 
action let's go ahead and build out a 
simple concise brand new feature inside 
of Pocket Pick there's a tweak that I've 
been wanting to make let me go ahead and 
open this up for you and just briefly 
explain Pocket Pick as engineers we 
reuse ideas patterns and code snippets 
all the time but keeping track of these 
can be hard pocket Pick is the solution 
to that problem this simplistic MCP 
server creates and stores all of your 
personal knowledgebased items right any 
code snippets files documentation that 
you want to reuse inside of a simple SQL 
light database here's all the tools you 
can see pocket add add add file find 
list so on and so forth the key change 
we want to make today is updating the 
add command if we look at the data types 
here to add new items to the pocket pick 
database we have text tags and database 
path right now the ID is automatically 
generated i want to improve the 
searching capabilities so that we can 
pass in ids when we're creating pocket 
items with pocket ad and pocket add file 
this just makes it super easy to run 
pocket git and pocket to file by ID how 
are we going to do that we're going to 
use these three essential directories 
with these directories we can move 
faster in this session and over time so 
let's go ahead i'm going to use of 
course claw code it's my favorite tool 
right now first thing we're going to do 
is prime so already having this command 
here I am saving time okay it's going to 
run as a tree get ignore it's going to 
give us a nice tree format for a coding 
tool to see you can see that working 
there let's go ahead and full screen 
this and then you can see I'm saying 
read the following files so this is all 
done that's all loaded we can type /cost 
and see exactly how much that costs we 
have 20 cents to get our agent primed 
now it's time to start doing real work 
how do we do real work we don't 
iteratively prompt that's the old 2024 
way of doing things we create concise 
agentic plans we're going to take this a 
step further with an emerging technique 
I like to call plan drafting the big 
difference here is that both myself and 
my AI coding tool are going to be part 
of drafting this plan so instead of 
writing the plan myself creating the 
file myself doing any of that work I'm 
going to have cloud code create the 
first draft of this plan create specs 
slash require ID a new feature ID stir 
inside of the and if I go back to the 
readme we can do a little bit of light 
planning here we want this inside of the 
pocket ad and the pocket break the plan 
into these 
sections write this plan I'm going to go 
ahead and use ultraink here it's 
definitely overkill but this is going to 
trigger claw code's reasoning capab 
capabilities what I'm doing here is 
having cloud code draft the first draft 
of this codebase i've briefly looked at 
this codebase i roughly remember the 
architecture and how it works but our 
agentic coding tools can quite literally 
read hundreds of times faster than we 
can it has a better understanding of 
this codebase than I already you know 
it's got the SQL like table structure it 
has all the commands right it can see UV 
piest right it knows about this codebase 
it can see the structure it you know 
quite literally knows more than I do 
right now about this codebase because we 
primed it properly and now it's going 
ahead it's going to write the first 
draft of this plan given everything that 
it knows okay and so what's the new kind 
of flow here the flow is build out the 
fundamentals of a plan have a draft get 
created by your AI coding tool and then 
iterate on the plan and then actually 
execute the plan so I'm going to go 
ahead and accept this you can see here 
we have a new spec created and let's go 
ahead and take a look at how this looks 
so this is the plan this is the first 
draft and you can see it's pretty good 
right it's detailing the problem uh very 
concisely right when users add items to 
the pocket pick database using the add 
and add file tools ids are automatically 
generated okay so problem statement and 
then solution statement this feature 
will modify these tools to require users 
to provide an ID as a mandatory 
parameter giving them more control and 
easier identification exactly so now 
we're reviewing right we're moving more 
and more every single day into a code 
reviewer into a plan reviewer we're 
becoming curators of information right 
curators of code ideas and then we're 
handing them off to our AI coding tools 
and you know the great part about this 
is that we're not iteratively updating 
our codebase putting it into bad 
temporary states we're operating solely 
in this file okay so I can look through 
everything you can see there it's got 
that new ID field in the server module 
right that's an essential change as well 
add functionality okay looks great 
server implementation looks awesome test 
changes you can see it knows where all 
these files are right this is the really 
important part right we're tying 
together that context prime we are 
skipping over our AI docs directory here 
because we don't really need it right if 
we were initially instantiating this 
codebase it might be more important all 
right so you can see here it has a self 
validation section that was built out 
and then updates to the readme so just 
by providing the right context with the 
context prime reusable prompt and by you 
know writing a you know pretty short 
what three sentence IDK rich prompt we 
activated the reasoning model 
capabilities we have a pretty great plan 
here i'm pretty sure this is going to 
take us um 80% of the way what I'm going 
to do here is delete some of the extra 
stuff here we have some recommendations 
i don't want the agent coding tool to 
build out any one of these extra 
optional ideas often times a coding 
tools language models if there's more 
text it'll try to create meaning it'll 
try to find patterns in it so I'm just 
going to go ahead and delete this schema 
stuff i'm going to add one tweak here 
check the other tests to see if they're 
using any ad functionality and update 
them to use this this is important and 
then I'm just going to you know open up 
this path to make it super clear on 
where to look this is honestly 
completely unnecessary but I just like 
to do this just to be crystal clear all 
right so great we have a great plan here 
let's go ahead and as a great practice I 
like to before you fire your plan off I 
always like to throw a commit in there 
let's make sure that we revert anything 
we did here this file here and then I'm 
going to commit the plan and then we're 
going to operate on the plan okay so I'm 
going to go ahead and say 
implement this file okay that's it cloud 
code has this new feature it's got a 
to-do list system this is a new emerging 
pattern inside of agentic coding tools 
where you effectively create a plan 
first and then you work through the plan 
this looks great i'm going to go ahead 
and go into aentic mode or yolo mode so 
now auto accepts are on and now our AI 
coding tool is just going to fly through 
this it's going to do all the work for 
us it is important to mention that I 
should have added the AID docs MCP 
server git repo mix integration at the 
same time I didn't need it because this 
functionality is already up and running 
there's really no um additional 
information here from thirdparty 
documentation that isn't already 
embedded inside of the codebase and so 
you know out of all the directories AI 
docs once you get up and running once 
you get your integration good is 
probably the least important but at the 
same time you know you don't want to 
underestimate the power of having you 
know a permanent knowledge base for your 
AI coding tools for your H engine coding 
tools that you can reference whenever 
you need them for your work especially 
when you're blowing up your context 
window over and over and over i also 
want to note that sometimes I have 
feature specific context priming so you 
know you can copy your prime and say you 
wanted to prime um add ID 
feature um and you know you can see here 
obviously our our great agent coding 
tools working through all the changes 
this is really cool love to see this 
with a feature specific context prime 
you can come in add this and you know 
say you wanted to add some specific file 
here right some feature specific file or 
files and then you know py or whatever 
and then you'd have multiple of these 
right so this lets you context prime on 
specific feature sets over and over and 
this is obviously better when you're 
working on larger changes often times 
you won't need it you'll just need your 
key prime method agent coding tool is 
working through these changes you can 
see that got implemented very quickly 
and it all comes back to writing a great 
plan you know I was very detailed very 
concise with my IDKs um you know shout 
out to all the principal a coding 
members that know your IDKs all my 
keywords here were packed with 
information i'm being very detailed 
about what I want you can see these 
items getting referenced the read me is 
getting updated now and it's all because 
I had my a coding tool see the right 
information and then I directed it to 
create a plan with that information with 
a specific structure set that I know 
works well right this is where your 
experience and your judgment and your 
taste comes in you have to know your 
codebase at some high level degree and 
then you can see there it's uh doing its 
self validation it's testing itself love 
to see that and you can see this is 
really important i added this at the end 
uh update other test files to include 
the ID parameter but you know just to 
summarize you know we we then created 
this great prompt that created the plan 
we're getting kind of meta we're now 
prompting our agents to write plans for 
us that we then tweak and iterate on and 
then we take that plan and we then hand 
it off to our AI coding tool or agent 
decoding tool again all right so you can 
see here it's working through these 
tests it needs to add that ID parameter 
to all of our ad commands to make sure 
that this works and again it's just 
chugging away for us this is why claw 
code is so good it does not stop working 
it doesn't ask you if you want to 
continue to do more work it just cooks 
right i I absolutely love this about 
Cloud Code it's how they've designed the 
model it's how they've designed the 
prompt i say it over and over but the 
Anthropic Cloud Co team is doing 
incredible work and it's clear to me 
that they use these tools you can see we 
missed one test case and test find you 
know we can always just open it up if we 
want to look side by side we can see 
that that functionality needs to get the 
ID that's now been added there is our 
concise summary of the work let's go 
ahead and take a look at this the most 
important thing here we have self 
validation on so I can pretty much be 
guaranteed I can be assured that 
everything here works right all the 
tests passed our hent coding tool is 
testing itself if we open up that spec 
you know we can see at the bottom here 
again just to mention it self- 
validation super important we gave it 
the command it needs to self-eleate just 
to highlight it again this is the big 
difference between agentic coding and AI 
coding i'm not just writing prompts that 
generate code okay i'm writing prompts 
that do engineering work that's building 
that's planning that's testing okay it's 
the whole development life cycle this is 
the power this is the capability you can 
unlock i aim to share and hand you 
valuable ideas like we've discussed here 
every single Monday you know exactly 
where to find me if you're not 
subscribed already definitely join the 
journey we're going to build living 
software that builds for us while we 
sleep and these three essential 
directories AI docs spec 
andclude this is how we scale up our 
engineering work this is how we pass off 
more work to our AI coding tools and now 
to our agentic coding tools all right 
make sure you like this video let the 
algorithm know that you're interested 
you can see all the tests pass here we 
don't really have to run this but of 
course we can updated pocket pick and 
what do we need to pass in here let's 
check the readme this one should be 
relatively simple yeah super simple 
let's go ahead and use this i'll say 
update there we go and yep database 
that's fine we can use the current 
working directory that's fine now I'm 
going to open up claude activate this 
MCP server you can see that's found 
there i'll hit yes and now I'm going to 
be really specific here with my prompts 
updated pocket pick i'll say pocket add 
and the ID here is going to be um let's 
just copy some random code let's say I 
want to you know remember the the 
MCP.json JSON format as a code snippet 
so I'll just copy that i'll say add i'll 
pass in the ID this is going to be MCP 
JSON and content is this okay so there 
we go i'm going to run our new Pocket ad 
using the updated Pocket Pick MCP server 
there it is updated Pocket Pick add 
there's the ID there's the text go ahead 
hit enter now we should get our new 
added item what's that tool called it's 
find we'll do this one right so pocket 
to file by ID this is the one that I 
like to run there and I'll say id and 
then you know that was 
mcp_json and then uh output file uh I'll 
say absolute path and get me you know 
mcpnew.json this is going to now search 
by our new ID and you can see we're 
running that updated pocket pick mcp 
server and we're going to output to this 
directory here let's go ahead and run 
this see how it works content 
successfully written let's open it up 
and bam you can see from our MCP server 
from that SQLite database this feature 
is running great and you can see there 
there's that new local database this is 
a great way to just store and reuse 
snippets link for this codebase is going 
to be in the description for you if 
you're interested so it's pretty 
incredible how quickly uh we were able 
to build that out here look at all the 
files that were just changed and 
remember what was done all these changes 
right very precise very surgical and 
it's all because of these essential 
files right these essential directories 
that let us scale what we can do inside 
of this codebase and every codebase i 
hope it's becoming clear as we spend 
more time together as you watch the 
channel it's really about the patterns 
it's about the principles it's about how 
you approach this new age of engineering 
don't get stuck on any specific tool and 
don't get stuck on ideas that don't 
scale across your work across your code 
bases and most importantly across time 
ai docs is your persistent knowledge 
base for your AI tooling specs is where 
you plan your work it's where you hand 
off more and more work to your compute 
to your agent coding tool remember great 
planning is great prompting and then we 
have doclude this is where we build 
reusable prompts we use across time in 
our codebase the most important prompt 
here to set up is the context prime 
prompt set up your agents so that they 
have the essential information to work 
concisely don't waste tokens giving them 
access to your entire codebase be 
precise be focused having too much 
context is just as bad as not having 
enough having too much context can 
confuse your agent having too little 
won't let them get the job done put 
these together and you can scale your 
engineering work further beyond thanks 
for watching stay focused and keep 
building